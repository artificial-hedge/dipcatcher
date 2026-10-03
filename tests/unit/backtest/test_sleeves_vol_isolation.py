"""SYNTHETIC regressions for symbol-local, causal sleeve volatility."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quant_fund.backtest.sleeves import _per_symbol_vol, funding_carry_weights

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _bars(prices: dict[str, list[float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"security_id": sid, "event_time": T0 + timedelta(hours=i), "close": price}
            for sid, series in prices.items()
            for i, price in enumerate(series)
        ]
    )


def _prices(n: int = 60) -> list[float]:
    return [10.0 + 0.2 * i + np.sin(i) for i in range(n)]


@pytest.mark.parametrize("window", [2, 4, 12, 48])
def test_panel_vol_matches_single_symbol_and_shifted_return_oracle(window: int) -> None:
    prices = _prices()
    bars = _bars({"A": [1000.0] * len(prices), "B": prices, "C": prices})
    actual = _per_symbol_vol(bars, window)
    min_samples = max(2, window // 4)
    returns = np.diff(np.log(prices))
    expected = [
        None
        if i - max(1, i - window) < min_samples
        else float(np.std(returns[max(0, i - window - 1) : i - 1], ddof=1))
        for i in range(len(prices))
    ]
    for sid in ("B", "C"):
        selected = actual.filter(pl.col("security_id") == sid)
        standalone = _per_symbol_vol(bars.filter(pl.col("security_id") == sid), window)
        assert_frame_equal(selected, standalone)
        for got, want in zip(selected["_vol"], expected, strict=True):
            if want is None:
                assert got is None
            else:
                assert got == pytest.approx(want)


def test_vol_is_invariant_to_input_order_and_symbol_sort_position() -> None:
    bars = _bars({"A": [1000.0] * 60, "B": _prices()})
    expected = _per_symbol_vol(bars, 12)
    assert_frame_equal(_per_symbol_vol(bars.sample(fraction=1, shuffle=True, seed=7), 12), expected)
    renamed = bars.with_columns(pl.col("security_id").replace({"A": "Z"}))
    actual = _per_symbol_vol(renamed, 12).with_columns(pl.col("security_id").replace({"Z": "A"}))
    assert_frame_equal(actual.sort(["security_id", "event_time"]), expected)


def test_preceding_symbol_future_close_cannot_change_earlier_vol() -> None:
    bars = _bars({"A": _prices(), "B": _prices()})
    changed = bars.with_columns(
        pl.when((pl.col("security_id") == "A") & (pl.col("event_time") == T0 + timedelta(hours=59)))
        .then(1e9)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert_frame_equal(_per_symbol_vol(changed, 12), _per_symbol_vol(bars, 12))


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_predecessor_close_does_not_change_other_symbol_vol(missing: float | None) -> None:
    prices: list[float | None] = list(_prices())
    # Missing prices retain existing within-symbol behavior but cannot spill over.
    prices[8] = missing
    bars = _bars({"A": [1000.0] * 59 + [missing], "B": prices})
    assert_frame_equal(
        _per_symbol_vol(bars, 12).filter(pl.col("security_id") == "B"),
        _per_symbol_vol(bars.filter(pl.col("security_id") == "B"), 12),
    )


def test_identical_funding_and_price_histories_emit_no_relative_carry() -> None:
    bars = _bars({"A": _prices(), "B": _prices()})
    funding = pl.DataFrame(
        {"security_id": ["A", "B"], "event_time": [T0, T0], "value": [-0.0001, -0.0001]}
    )
    # Identical raw signals must cancel under cross-sectional median centering.
    assert funding_carry_weights(bars, funding, vol_window=12).is_empty()
