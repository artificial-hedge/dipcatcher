"""Kakushadze residual mean-reversion sleeve: directionality + causality."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest import sleeves
from quant_fund.backtest.sleeves import residual_mr_weights

N_BARS = 260
NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _bars(seed: int = 0, shock_name: str | None = None, shock_bar: int = 200, shock: float = 0.0):
    """Factor + idiosyncratic book; optional one-bar shock to ``shock_name``."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    factor = rng.normal(0.0, 0.004, size=N_BARS)
    for i, sid in enumerate(NAMES):
        beta = 0.5 + 0.2 * i
        idio = rng.normal(0.0, 0.002, size=N_BARS)
        r = beta * factor + idio
        if shock_name is not None and sid == shock_name:
            r[shock_bar] += shock
        px = 100.0 * np.exp(np.cumsum(r))
        for t in range(N_BARS):
            dt = start + timedelta(hours=t)
            c = px[t]
            o = px[t - 1] if t else c / np.exp(r[0])
            rows.append(
                {
                    "event_time": dt,
                    "security_id": sid,
                    "open": o,
                    "high": max(o, c) * 1.001,
                    "low": min(o, c) * 0.999,
                    "close": c,
                    "volume": 1e5,
                }
            )
    return pl.DataFrame(rows)


def test_emits_capped_dollar_neutral_weights() -> None:
    w = residual_mr_weights(_bars())
    assert set(w.columns) == {"event_time", "security_id", "target_weight"}
    assert w["target_weight"].abs().max() <= 0.05 + 1e-12
    gross = w.group_by("event_time").agg(pl.col("target_weight").abs().sum().alias("g"))
    assert gross["g"].max() <= 1.0 + 1e-9
    # Demeaned: median ≈ 0 wherever enough names emitted.
    med = w.group_by("event_time").agg(pl.col("target_weight").median().alias("m"))
    assert med["m"].abs().max() < 0.05


def test_residual_shock_flips_weight_sign() -> None:
    """A one-bar positive residual shock must produce a short weight next bars."""
    bars = _bars(shock_name="CCC", shock_bar=200, shock=+0.05)
    w = residual_mr_weights(bars)
    shock_t = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=200)
    after = w.filter((pl.col("security_id") == "CCC") & (pl.col("event_time") > shock_t)).sort(
        "event_time"
    )
    assert after.height > 0
    assert after["target_weight"][0] < 0.0
    # Symmetric: a negative shock goes long.
    bars_dn = _bars(shock_name="CCC", shock_bar=200, shock=-0.05)
    w_dn = residual_mr_weights(bars_dn)
    after_dn = w_dn.filter(
        (pl.col("security_id") == "CCC") & (pl.col("event_time") > shock_t)
    ).sort("event_time")
    assert after_dn["target_weight"][0] > 0.0


def test_causality_tail_perturbation() -> None:
    """Rewriting the tail of the book leaves every earlier weight identical."""
    bars = _bars()
    cut = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=180)
    mutated = bars.with_columns(
        pl.when(pl.col("event_time") > cut)
        .then(pl.col("close") * 1.5)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    w_base = residual_mr_weights(bars).filter(pl.col("event_time") <= cut)
    w_mut = residual_mr_weights(mutated).filter(pl.col("event_time") <= cut)
    assert w_base.equals(w_mut)


def test_determinism() -> None:
    assert residual_mr_weights(_bars(seed=3)).equals(residual_mr_weights(_bars(seed=3)))


def test_validation() -> None:
    bars = _bars()
    with pytest.raises(ValueError, match="factor_window"):
        residual_mr_weights(bars, factor_window=1)
    with pytest.raises(ValueError, match="z_clip"):
        residual_mr_weights(bars, z_clip=0.0)
    with pytest.raises(ValueError, match="missing columns"):
        residual_mr_weights(bars.drop("close"))


@pytest.fixture
def residual_frame(monkeypatch: pytest.MonkeyPatch):
    """Inspect the actual estimator before clipping can hide beta errors."""
    monkeypatch.setattr(sleeves, "_cap_and_emit", lambda frame, *args, **kwargs: frame)
    return residual_mr_weights


@pytest.mark.parametrize("window", [2, 5, 16])
def test_beta_recovers_exact_synthetic_factor(residual_frame, window: int) -> None:
    """SYNTHETIC oracle: r_i = beta_i * m, with mean(beta_i) = 1."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    # Alternating signs keep even the two-observation variance well conditioned.
    factor = np.random.default_rng(81).uniform(0.005, 0.02, 64)
    factor[::2] *= -1
    betas = [-0.5, 0.5, 1.0, 1.5, 2.5]
    rows = []
    for sid, beta in zip(NAMES, betas, strict=True):
        for t, close in enumerate(100.0 * np.exp(np.cumsum(beta * factor))):
            rows.append((start + timedelta(hours=t), sid, close))
    bars = pl.DataFrame(rows, schema=["event_time", "security_id", "close"], orient="row")
    frame = residual_frame(bars, factor_window=window)
    for sid, beta in zip(NAMES, betas, strict=True):
        group = frame.filter(pl.col("security_id") == sid)
        # Initial log-diff is null; the full window and one-bar lag are required.
        assert group["_beta"][: window + 1].null_count() == window + 1
        np.testing.assert_allclose(group["_beta"][window + 1 :], beta, atol=1e-10)
        np.testing.assert_allclose(group["_resid_now"][window + 1 :], 0.0, atol=1e-12)


@pytest.mark.parametrize("window", [2, 5, 16])
def test_beta_matches_trailing_ols_with_nulls_and_staggered_names(
    residual_frame, window: int
) -> None:
    """SYNTHETIC rolling OLS oracle, including full-window null propagation."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    bars = (
        _bars(seed=13)
        .filter(
            (pl.col("security_id") != "EEE") | (pl.col("event_time") >= start + timedelta(hours=10))
        )
        .with_columns(
            pl.when(
                (pl.col("security_id") == "BBB")
                & (pl.col("event_time") == start + timedelta(hours=30))
            )
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
    )
    frame = residual_frame(bars.sample(fraction=1.0, shuffle=True, seed=4), factor_window=window)
    for group in frame.partition_by("security_id"):
        returns = group["_r"].to_numpy()
        factor = group["_mkt"].to_numpy()
        expected = np.full(group.height, np.nan)
        for t in range(window, group.height):
            x, y = factor[t - window : t], returns[t - window : t]
            if np.isfinite(x).all() and np.isfinite(y).all():
                # An intercept gives the centered covariance/variance slope.
                expected[t] = np.linalg.lstsq(np.column_stack([np.ones(window), x]), y, rcond=None)[
                    0
                ][1]
        actual = group["_beta"].to_numpy()
        np.testing.assert_array_equal(group["_beta"].is_null(), np.isnan(expected))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_current_bar_mutation_cannot_change_its_weight() -> None:
    """The beta fix must preserve the strictly-prior-bar signal contract."""
    bars = _bars(seed=23)
    cut = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=180)
    mutated = bars.with_columns(
        pl.when((pl.col("event_time") >= cut) & (pl.col("security_id") == "AAA"))
        .then(pl.col("close") * 1.5)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    before = residual_mr_weights(bars).filter(pl.col("event_time") <= cut)
    after = residual_mr_weights(mutated).filter(pl.col("event_time") <= cut)
    assert before.filter(pl.col("event_time") == cut).height > 0
    assert before.equals(after)
