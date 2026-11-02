"""Coverage: BookRiskOverlay parameter guards and degenerate history branches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.risk.overlay import BookRiskOverlay


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vol_target": 0.0}, "vol_target must be finite and positive"),
        ({"vol_target": float("nan")}, "vol_target must be finite and positive"),
        ({"dd_limit": 0.0}, r"dd_limit must be in \(0, 1\)"),
        ({"dd_limit": 1.5}, r"dd_limit must be in \(0, 1\)"),
        ({"dd_limit": float("inf")}, r"dd_limit must be in \(0, 1\)"),
        ({"es_limit": 0.0}, "es_limit must be finite and positive"),
        ({"es_limit": float("nan")}, "es_limit must be finite and positive"),
        ({"kelly_fraction": -0.1}, "kelly_fraction must be finite and non-negative"),
        ({"kelly_fraction": float("nan")}, "kelly_fraction must be finite and non-negative"),
        ({"crc_alpha": -0.01}, r"crc_alpha must be in \[0, 1\)"),
        ({"crc_alpha": 1.0}, r"crc_alpha must be in \[0, 1\)"),
        ({"crc_alpha": float("nan")}, r"crc_alpha must be in \[0, 1\)"),
    ],
)
def test_ctor_rejects_bad_parameters(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BookRiskOverlay(**kwargs)


def test_short_lookbacks_are_floored_at_eight() -> None:
    overlay = BookRiskOverlay(lookback=1, kelly_lookback=0, crc_lookback=-5)
    assert overlay.lookback == 8
    assert overlay.kelly_lookback == 8
    assert overlay.crc_lookback == 8


def test_preview_scale_before_any_observation_uses_prior() -> None:
    overlay = BookRiskOverlay(vol_target=0.025)
    # Single observation: no return window yet -> conservative 20% vol prior.
    overlay.observe(100.0)
    assert overlay.preview_scale() == pytest.approx(min(1.0, 0.025 / 0.20))
    assert overlay.n_scaled == 1
    assert overlay.n_halt == 0


def test_non_positive_nav_permanently_ruins_book() -> None:
    overlay = BookRiskOverlay()
    overlay.observe(100.0)
    overlay.observe(-1.0)
    assert overlay.ruined is True
    assert overlay.navs == [100.0]
    assert overlay.preview_scale() == 0.0
    assert overlay.n_halt == 1


def test_missing_return_window_returns_sized_prior() -> None:
    overlay = BookRiskOverlay(vol_target=0.025)
    for nav in (100.0, 100.5, 101.0, 100.8):
        overlay.observe(nav)
    # Fewer than 8 returns: fall back to the 20% vol prior, not full size.
    assert overlay.preview_scale() == pytest.approx(min(1.0, 0.025 / 0.20))


def test_nonpositive_history_returns_zero_scale() -> None:
    overlay = BookRiskOverlay()
    # observe() refuses such values; a corrupted cache must still fail closed.
    overlay.navs.extend([100.0, 0.0])
    assert overlay.preview_scale() == 0.0
    assert overlay.n_halt == 1


def test_nonfinite_history_keeps_conservative_prior() -> None:
    overlay = BookRiskOverlay(vol_target=0.025)
    overlay.navs.extend([100.0, 101.0, 100.0, 99.0, 100.0, 101.0, 100.0, 99.0, 100.0, math.nan])
    scale = overlay.preview_scale()
    assert scale == pytest.approx(min(1.0, 0.025 / 0.20))


def test_drawdown_budget_exhausted_halts() -> None:
    overlay = BookRiskOverlay(dd_limit=0.05)
    # Peak 100 then a 4.8% drawdown leaves less than the 50 bp cushion.
    for nav in (100.0, 95.2):
        overlay.observe(nav)
    assert overlay.preview_scale() == 0.0
    assert overlay.n_halt == 1


def test_negative_kelly_mean_halts_at_full_lookback() -> None:
    overlay = BookRiskOverlay(kelly_fraction=0.25, kelly_lookback=8)
    navs = [100.0]
    for _ in range(12):
        navs.append(navs[-1] * 0.999)
    for nav in navs:
        overlay.observe(nav)
    # Skewed down-drift: fractional Kelly must clip to zero, never flip sign.
    assert overlay.preview_scale() == 0.0


def test_snapshot_defaults_before_any_scale() -> None:
    snap = BookRiskOverlay().snapshot()
    assert snap["n_observe"] == 0
    assert snap["last_scale"] == 1.0
    assert snap["mean_scale"] == 1.0
    assert snap["ruined"] is False
    assert snap["research_only"] is True
    assert snap["execution_claim"] == "paper_backtest"


def test_snapshot_after_scaling_reports_state() -> None:
    overlay = BookRiskOverlay(vol_target=0.01)
    for nav in np.linspace(100.0, 101.0, 20):
        overlay.observe(float(nav))
    overlay.preview_scale()
    snap = overlay.snapshot()
    assert snap["n_observe"] == 20
    assert snap["last_scale"] == overlay.scales[-1]
    assert snap["mean_scale"] == pytest.approx(float(np.mean(overlay.scales)))
    assert snap["sources"] == ("lprtk/pyRisk", "GianMarcoOddo/pyriskmgmt")
