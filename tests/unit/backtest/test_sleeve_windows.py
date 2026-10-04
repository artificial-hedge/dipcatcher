"""SYNTHETIC window contracts and rolling warmup oracles; no market evidence."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quant_fund.backtest import sleeves

T0 = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def panel():
    rng = np.random.default_rng(31)
    rows = []
    for sid in ("A", "B", "C"):
        for i, price in enumerate(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))):
            rows.append((T0 + timedelta(hours=i), sid, price, price, price * 1.01, price * 0.99))
    return pl.DataFrame(
        rows, schema=["event_time", "security_id", "close", "open", "high", "low"], orient="row"
    )


@pytest.fixture
def funding(panel):
    return panel.select("event_time", "security_id").with_columns(
        pl.when(pl.col("security_id") == "A").then(0.01).otherwise(-0.01).alias("value")
    )


# Each public window, including shared helper callers, must reject without coercion.
CASES = [
    (sleeves.funding_carry_weights, "lookback_events", 1, True),
    (sleeves.funding_spike_fade_weights, "lookback_events", 10, True),
    (sleeves.basis_carry_weights, "lookback_events", 1, True),
    (sleeves.basis_carry_hysteresis_weights, "lookback_events", 1, True),
    (sleeves.basis_carry_hysteresis_weights, "vol_lookback", 5, True),
    (sleeves.funding_carry_weights, "vol_window", 2, True),
    (sleeves.funding_spike_fade_weights, "vol_window", 2, True),
    (sleeves.cross_sectional_momentum_weights, "vol_window", 2, False),
    (sleeves.slow_trend_weights, "vol_window", 2, False),
    (sleeves.cross_sectional_momentum_weights, "lookback_bars", 1, False),
    (sleeves.cross_sectional_momentum_weights, "skip_bars", 0, False),
    (sleeves.slow_trend_weights, "fast_bars", 1, False),
    (sleeves.slow_trend_weights, "slow_bars", 2, False),
    (sleeves.sweep_reclaim_weights, "lookback", 2, False),
    (sleeves.sweep_reclaim_weights, "hold_bars", 1, False),
    (sleeves.residual_mr_weights, "factor_window", 2, False),
    (sleeves.residual_mr_weights, "z_window", 2, False),
    (sleeves.residual_mr_weights, "reversal_window", 2, False),
]


@pytest.mark.parametrize("func,label,minimum,needs_funding", CASES)
@pytest.mark.parametrize("bad", [True, False, 2.0, 2.5, "2", -1, float("nan"), float("inf")])
def test_invalid_window_types_and_negative_values(
    panel, funding, func, label, minimum, needs_funding, bad
):
    args = (panel, funding) if needs_funding else (panel,)
    with pytest.raises(ValueError, match=label):
        func(*args, **{label: bad})


@pytest.mark.parametrize("func,label,minimum,needs_funding", CASES)
def test_window_lower_bound(panel, funding, func, label, minimum, needs_funding):
    args = (panel, funding) if needs_funding else (panel,)
    for bad in range(minimum):
        with pytest.raises(ValueError, match=label):
            func(*args, **{label: bad})


@pytest.mark.parametrize("z_window", [2, 3, 4, 5, 48])
def test_residual_sigma_warmup_matches_sample_std(panel, monkeypatch, z_window):
    monkeypatch.setattr(sleeves, "_cap_and_emit", lambda frame, *args, **kwargs: frame)
    frame = sleeves.residual_mr_weights(
        panel, factor_window=2, z_window=z_window, reversal_window=2
    )
    warmup = min(z_window, max(4, z_window // 4))
    for group in frame.partition_by("security_id"):
        resid = group["_resid_now"].to_numpy()
        expected = np.full(group.height, np.nan)
        for t in range(group.height):
            past = resid[max(0, t - z_window - 1) : max(0, t - 1)]
            past = past[np.isfinite(past)]
            if len(past) >= warmup:
                expected[t] = np.std(past, ddof=1)
        np.testing.assert_allclose(group["_resid_sigma"].to_numpy(), expected, atol=1e-12)
        assert group["_resid_sigma"][: warmup + 4].null_count() == warmup + 4
        assert group["_resid_sigma"][warmup + 4] is not None


@pytest.mark.parametrize("window", [2, 3, 4])
def test_small_residual_windows_emit_causal_finite_weights(panel, window):
    kwargs = {"factor_window": 2, "z_window": window, "reversal_window": 2}
    expected = sleeves.residual_mr_weights(panel, **kwargs)
    assert expected.height > 0
    assert expected["target_weight"].is_finite().all()
    assert expected["event_time"].min() == T0 + timedelta(hours=window + 4)
    cut = T0 + timedelta(hours=40)
    changed = panel.with_columns(
        pl.when((pl.col("event_time") >= cut) & (pl.col("security_id") == "A"))
        .then(pl.col("close") * 2)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert_frame_equal(
        expected.filter(pl.col("event_time") <= cut),
        sleeves.residual_mr_weights(changed, **kwargs).filter(pl.col("event_time") <= cut),
    )


def test_legitimate_minimum_windows(panel, funding):
    assert sleeves.funding_carry_weights(panel, funding, lookback_events=1, vol_window=2).height
    assert sleeves.basis_carry_weights(panel, funding, lookback_events=1).height
    assert sleeves.basis_carry_hysteresis_weights(panel, funding, vol_lookback=5).height
    assert sleeves.cross_sectional_momentum_weights(
        panel, lookback_bars=1, skip_bars=0, vol_window=2
    ).height
    assert sleeves.slow_trend_weights(panel, fast_bars=1, slow_bars=2, vol_window=2).height
    assert sleeves.sweep_reclaim_weights(panel, lookback=2, hold_bars=1).height


def test_spike_fade_minimum_window_warms_up_at_tenth_event(panel, funding):
    funding = funding.with_columns(
        pl.when((pl.col("security_id") == "A") & (pl.col("event_time") == T0 + timedelta(hours=9)))
        .then(0.5)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    weights = sleeves.funding_spike_fade_weights(panel, funding, lookback_events=10, vol_window=2)
    assert weights.height > 0
    assert weights["event_time"].min() == T0 + timedelta(hours=9)


@pytest.mark.parametrize("event_bar", [4, 5, 6])
def test_hysteresis_five_return_warmup(panel, event_bar):
    bars = panel.filter(pl.col("security_id") == "A")
    funding = bars.filter(pl.col("event_time") == T0 + timedelta(hours=event_bar)).select(
        "event_time", "security_id", pl.lit(0.01).alias("value")
    )
    weights = sleeves.basis_carry_hysteresis_weights(
        bars, funding, vol_lookback=5, vol_ref=0.001, name_weight=0.08
    )
    prices = bars["close"].to_numpy()
    returns = prices[1:] / prices[:-1] - 1
    # Existing policy leaves weights unscaled until five returns are available.
    expected = 0.08
    if event_bar >= 5:
        expected *= min(1, 0.001 / np.std(returns[event_bar - 5 : event_bar], ddof=1))
    assert weights["target_weight"][0] == pytest.approx(expected)
