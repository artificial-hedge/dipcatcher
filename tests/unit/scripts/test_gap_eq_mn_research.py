"""Tests for ``scripts/eq_mn_research.py``.

This is the market-neutral *screening* harness: a fast approximation of the
real backtest used to shortlist signals before they are confirmed through
``run_backtest``. The properties that matter for a screening tool are
structural, not performance:

* every signal transform is **causal** — a row at time *t* is a function of
  data up to and including the close of *t*, never of the future;
* the transform functions are well behaved at the edges (warm-up rows are
  NaN, a zero-variance cross-section does not divide by zero);
* the book construction respects its declared dollar-neutrality, gross and
  per-name cap constraints.

The metrics this harness computes are the harness's own screening statistics,
not research results: nothing here is a market-evidence claim.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "eq_mn_research.py"
_spec = importlib.util.spec_from_file_location("gap_eq_mn_research", _SCRIPT_PATH)
assert _spec and _spec.loader, "eq_mn_research.py must be importable"
mod = importlib.util.module_from_spec(_spec)
sys.modules["gap_eq_mn_research"] = mod
_spec.loader.exec_module(mod)


def _panel(t: int = 60, n: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    tr = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(t, n)), axis=0)
    open_ = tr * (1.0 + rng.normal(0.0, 0.002, size=(t, n)))
    close = tr * (1.0 + rng.normal(0.0, 0.002, size=(t, n)))
    volu = rng.uniform(1e5, 1e6, size=(t, n))
    sectors = np.array([f"s{i % 3}" for i in range(n)])
    return {"tr": tr, "open": open_, "close": close, "volu": volu, "sectors": sectors}


# --------------------------------------------------------------------------- #
# zscore_xs
# --------------------------------------------------------------------------- #


def test_cross_sectional_zscores_are_clipped() -> None:
    x = np.array([[1.0, 2.0, 3.0, 100.0]])
    z = mod.zscore_xs(x)
    assert np.all(np.abs(z) <= 3.0)


def test_a_constant_cross_section_yields_nan_not_infinity() -> None:
    """Zero dispersion must not divide by zero."""
    z = mod.zscore_xs(np.ones((3, 4)))
    assert np.all(np.isnan(z))


def test_zscores_are_centred_across_the_cross_section() -> None:
    rng = np.random.default_rng(1)
    z = mod.zscore_xs(rng.normal(size=(5, 8)))
    assert np.allclose(np.nanmean(z, axis=1), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# demean_groups
# --------------------------------------------------------------------------- #


def test_group_demeaning_centres_each_group() -> None:
    z = np.array([[1.0, 2.0, 10.0, 20.0]])
    groups = np.array(["a", "a", "b", "b"])
    out = mod.demean_groups(z, groups)
    assert out[0, 0] == pytest.approx(-0.5)
    assert out[0, 1] == pytest.approx(0.5)
    assert out[0, 2] == pytest.approx(-5.0)
    assert out[0, 3] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# rolling_sum
# --------------------------------------------------------------------------- #


def test_rolling_sum_has_a_warm_up_and_then_sums_the_window() -> None:
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = mod.rolling_sum(x, 2)
    assert np.isnan(out[0, 0])
    assert np.isnan(out[1, 0])
    # The window is the k rows ending at t (inclusive).
    assert out[2, 0] == pytest.approx(5.0)
    assert out[3, 0] == pytest.approx(7.0)


def test_rolling_sum_drops_a_row_with_a_nan_inside_the_window() -> None:
    x = np.array([[1.0], [np.nan], [3.0], [4.0]])
    out = mod.rolling_sum(x, 2)
    assert np.isnan(out[2, 0])
    assert out[3, 0] == pytest.approx(7.0)


# --------------------------------------------------------------------------- #
# ewma
# --------------------------------------------------------------------------- #


def test_ewma_seeds_on_the_first_row_and_decays_afterwards() -> None:
    x = np.array([[0.0], [1.0]])
    out = mod.ewma(x, hl=1.0)
    assert out[0, 0] == pytest.approx(0.0)
    alpha = 1 - 0.5 ** (1 / 1.0)
    assert out[1, 0] == pytest.approx(alpha * 1.0)


def test_ewma_propagates_nan_rows() -> None:
    x = np.array([[1.0], [np.nan], [2.0]])
    out = mod.ewma(x, hl=2.0)
    assert np.isnan(out[1, 0])


# --------------------------------------------------------------------------- #
# Weight construction
# --------------------------------------------------------------------------- #


def test_weights_are_dollar_neutral() -> None:
    z = np.array([[1.0, 2.0, 3.0, 4.0]])
    w = mod.to_weights(z, gross=2.0, cap=1.0)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)


def test_the_gross_target_is_respected() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(size=(3, 10))
    w = mod.to_weights(z, gross=2.0, cap=1.0)
    assert np.allclose(np.abs(w).sum(axis=1), 2.0)


def test_the_per_name_cap_is_respected() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(size=(2, 8)) * 100.0
    w = mod.to_weights(z, gross=2.0, cap=0.03)
    assert np.all(np.abs(w) <= 0.03 + 1e-12)


def test_an_all_zero_cross_section_produces_zero_weights() -> None:
    z = np.zeros((2, 5))
    w = mod.to_weights(z, gross=2.0, cap=0.03)
    assert np.allclose(w, 0.0)


def test_sector_neutral_weights_are_dollar_neutral_within_a_sector() -> None:
    sectors = np.array(["a", "a", "b", "b"])
    z = np.array([[1.0, 3.0, 1.0, 3.0]])
    w = mod.to_weights(z, gross=2.0, cap=1.0, sector_neutral=np.ones(1), sectors=sectors)
    assert w[0, 0] + w[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert w[0, 2] + w[0, 3] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Smoothing and vol targeting
# --------------------------------------------------------------------------- #


def test_smoothing_lags_the_signal() -> None:
    w = np.array([[1.0], [0.0]])
    out = mod.smooth_weights(w, hl=1.0)
    alpha = 1 - 0.5 ** (1 / 1.0)
    assert out[0, 0] == pytest.approx(alpha * 1.0)
    assert out[1, 0] == pytest.approx(alpha * 0.0 + (1 - alpha) * alpha)


def test_vol_targeting_leaves_the_warm_up_window_untouched() -> None:
    w = np.ones((50, 2))
    tr = np.tile(np.linspace(100.0, 110.0, 50)[:, None], (1, 2))
    out = mod.vol_target_weights(w, tr, target_vol=0.10, lb=40)
    assert np.allclose(out[:41], w[:41])


def test_vol_targeting_never_exceeds_the_leverage_cap() -> None:
    rng = np.random.default_rng(4)
    tr = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.05, size=(120, 3)), axis=0)
    w = rng.normal(size=(120, 3))
    out = mod.vol_target_weights(w, tr, target_vol=0.10, lb=40, max_lev=2.0)
    assert np.all(np.abs(out) <= 2.0 * np.abs(w) + 1e-12)


def test_vol_targeting_is_causal() -> None:
    """Changing a future row must not change an earlier scaled row."""
    rng = np.random.default_rng(5)
    tr = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=(100, 2)), axis=0)
    w = rng.normal(size=(100, 2))
    base = mod.vol_target_weights(w, tr, target_vol=0.1, lb=40)
    moved = tr.copy()
    moved[80:] *= 1.5
    other = mod.vol_target_weights(w, moved, target_vol=0.1, lb=40)
    assert np.allclose(base[:80], other[:80])


# --------------------------------------------------------------------------- #
# Simulation and metrics
# --------------------------------------------------------------------------- #


def test_the_fast_backtest_returns_one_nav_point_per_bar() -> None:
    panel = _panel(t=40, n=4)
    weights = np.zeros((40, 4))
    res = mod.fast_bt(panel["open"], panel["tr"], panel["volu"], weights)
    assert res["nav"].shape == (40,)
    assert res["turnover"].shape == (40,)
    assert res["cost_total"] >= 0.0


def test_a_flat_book_pays_no_market_impact_cost() -> None:
    panel = _panel(t=30, n=3)
    weights = np.zeros((30, 3))
    res = mod.fast_bt(panel["open"], panel["tr"], panel["volu"], weights)
    assert res["cost_total"] == pytest.approx(0.0)


def test_book_metrics_report_the_sample_size_and_the_total_change() -> None:
    nav = np.array([1.0, 1.1, 1.2, 1.3])
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    metrics = mod.book_metrics(nav, dates)
    assert metrics["n"] == 3
    assert metrics["total"] == pytest.approx(0.3, rel=1e-6)
    assert metrics["mdd"] <= 0.0
    assert metrics["t0"] == "2024-01-01"
    assert metrics["t1"] == "2024-01-04"


def test_a_too_short_history_returns_nan_metrics() -> None:
    metrics = mod.book_metrics(np.array([1.0, 1.0]), ["2024-01-01", "2024-01-02"])
    assert "n" not in metrics
    assert np.isnan(metrics["total"])
    assert np.isnan(metrics["mdd"])


# --------------------------------------------------------------------------- #
# Signal library
# --------------------------------------------------------------------------- #


def test_the_signal_library_returns_one_matrix_per_signal() -> None:
    panel = _panel(t=300, n=6)
    sigs = mod.signals(panel["tr"], panel["open"], panel["close"], panel["volu"], panel["sectors"])
    assert sigs
    for name, signal in sigs.items():
        assert signal.shape == (300, 6), name


def test_the_signal_library_is_causal() -> None:
    """Rewriting the last rows must not move an earlier signal value."""
    panel = _panel(t=300, n=6)
    base = mod.signals(panel["tr"], panel["open"], panel["close"], panel["volu"], panel["sectors"])
    moved = {key: value.copy() for key, value in panel.items()}
    moved["tr"][-5:] *= 3.0
    moved["open"][-5:] *= 3.0
    moved["close"][-5:] *= 3.0
    other = mod.signals(moved["tr"], moved["open"], moved["close"], moved["volu"], moved["sectors"])
    for name in base:
        assert np.allclose(base[name][:-5], other[name][:-5], equal_nan=True), name


def test_the_reversal_signal_is_the_negative_of_the_return_zscore() -> None:
    """Higher ``rev5`` must mean *more* attractive (long), so it is negated."""
    panel = _panel(t=120, n=5)
    sigs = mod.signals(panel["tr"], panel["open"], panel["close"], panel["volu"], panel["sectors"])
    r1 = np.full_like(panel["tr"], np.nan)
    r1[1:] = panel["tr"][1:] / panel["tr"][:-1] - 1
    lr1 = np.log1p(np.clip(r1, -0.99, 10))
    expected = -mod.zscore_xs(mod.rolling_sum(lr1, 5))
    assert np.allclose(sigs["rev5"], expected, equal_nan=True)
