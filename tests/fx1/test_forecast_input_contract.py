"""Offline regressions for forecast feature and resampling input contracts."""

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from fx1.forecast.features import OhlcvFeaturePipeline, resample_ohlcv
from quant_fund.schemas.errors import PointInTimeError


def _bars() -> pl.DataFrame:
    start = datetime(2020, 1, 2, 9, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "security_id": sid,
                "event_time": start + timedelta(hours=i),
                "available_time": start + timedelta(hours=i),
                "open": 10.0 + i,
                "high": 12.0 + i,
                "low": 9.0 + i,
                "close": 11.0 + i,
                "volume": 5.0,
            }
            for sid in ("AAA", "BBB")
            for i in range(3)
        ]
    )


@pytest.mark.parametrize("conflicting", [False, True])
def test_resample_rejects_duplicate_bar_keys(conflicting: bool) -> None:
    bars = _bars()
    duplicate = bars.head(1)
    if conflicting:
        duplicate = duplicate.with_columns(pl.lit(12.0).alias("close"))
    bars = pl.concat([bars, duplicate])
    with pytest.raises(PointInTimeError, match="duplicate"):
        resample_ohlcv(bars, "1d")


def test_resample_preserves_distinct_security_keys_and_valid_aggregation() -> None:
    bars = _bars()
    expected = resample_ohlcv(bars, "1d")
    assert expected.equals(resample_ohlcv(bars.reverse(), "1d"))
    assert expected["security_id"].to_list() == ["AAA", "BBB"]
    for row in expected.iter_rows(named=True):
        assert row["open"] == 10.0
        assert row["high"] == 14.0
        assert row["low"] == 9.0
        assert row["close"] == 13.0
        assert row["volume"] == 15.0
        assert row["event_time"] == datetime(2020, 1, 2, 11, tzinfo=UTC)
        assert row["available_time"] == row["event_time"]


@pytest.mark.parametrize("value", [1.5, 2.0, "2", True, None, float("nan"), float("inf")])
def test_lookbacks_reject_noninteger_values(value: Any) -> None:
    with pytest.raises(ValueError, match="lookbacks must be positive integers"):
        OhlcvFeaturePipeline([value], 2)


@pytest.mark.parametrize("value", [2.5, 2.0, "2", True, None, float("nan"), float("inf")])
def test_vol_window_rejects_noninteger_values(value: Any) -> None:
    with pytest.raises(ValueError, match="vol_window must be an integer >= 2"):
        OhlcvFeaturePipeline([1], value)


def test_integer_windows_are_sorted_and_preserved() -> None:
    pipeline = OhlcvFeaturePipeline([np.int64(5), 1, 3], np.int64(4))
    assert pipeline.lookbacks == (1, 3, 5)
    assert pipeline.vol_window == 4
    assert pipeline.feature_columns() == ["ret_1", "mom_3", "mom_5", "vol_4"]


@pytest.mark.parametrize("lookbacks", [[], [0], [-1], [1, 1]])
def test_lookback_bounds_and_uniqueness_still_enforced(lookbacks: list[int]) -> None:
    with pytest.raises(ValueError, match="lookbacks"):
        OhlcvFeaturePipeline(lookbacks, 2)


@pytest.mark.parametrize("window", [-1, 0, 1])
def test_vol_window_lower_bound_still_enforced(window: int) -> None:
    with pytest.raises(ValueError, match="vol_window"):
        OhlcvFeaturePipeline([1], window)
