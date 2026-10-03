"""SYNTHETIC checks for source availability before feature transformations."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.features.engine import build_features
from quant_fund.schemas.errors import PointInTimeError


def _bars(first_close: float = 50.0) -> pl.DataFrame:
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(4)]
    prices = [first_close, 100.0, 110.0, 120.0]
    return pl.DataFrame(
        {
            "security_id": ["A"] * 4,
            "event_time": times,
            "available_time": times,
            "close_total_return": prices,
            "close": prices,
            "volume": [1_000.0] * 4,
            "open_split_adjusted": prices,
            "close_split_adjusted": prices,
            "high_split_adjusted": [p + 1 for p in prices],
            "low_split_adjusted": [p - 1 for p in prices],
        }
    )


def _later_membership(bars: pl.DataFrame) -> pl.DataFrame:
    return bars.select("security_id", pl.col("event_time").alias("asof")).slice(1)


@pytest.mark.parametrize("unavailable", ["late", "null"])
@pytest.mark.parametrize("first_close", [50.0, 200.0])
def test_unavailable_nonmember_history_fails_closed(unavailable: str, first_close: float) -> None:
    bars = _bars(first_close)
    first = bars["event_time"][0]
    published = bars["event_time"][-1] + timedelta(days=1) if unavailable == "late" else None
    bars = bars.with_columns(
        pl.when(pl.col("event_time") == first)
        .then(pl.lit(published, dtype=bars.schema["available_time"]))
        .otherwise(pl.col("available_time"))
        .alias("available_time"),
    )
    # Before the guard, the excluded first row changes Jan 2 ret_1 from 1.0
    # to -0.5 despite remaining unpublished on every emitted decision date.
    with pytest.raises(PointInTimeError, match="available_time"):
        build_features(bars, AppConfig(), membership=_later_membership(bars))


def test_null_availability_cannot_hide_behind_cross_sectional_max() -> None:
    bars = _bars()
    unknown = bars.with_columns(
        pl.lit("B").alias("security_id"),
        pl.lit(None, dtype=bars.schema["available_time"]).alias("available_time"),
    )
    with pytest.raises(PointInTimeError, match="available_time"):
        build_features(pl.concat([bars, unknown]), AppConfig())


def test_input_feature_metadata_cannot_override_source_clock() -> None:
    bars = _bars().with_columns(
        (pl.col("event_time") + timedelta(days=20)).alias("available_time"),
        (pl.col("event_time") + timedelta(days=30)).alias("decision_time"),
        pl.col("event_time").alias("max_source_available_time"),
    )
    with pytest.raises(PointInTimeError, match="available_time"):
        build_features(bars, AppConfig(), membership=_later_membership(bars))


@pytest.mark.parametrize("early", [False, True])
def test_timely_nonmember_history_remains_usable(early: bool) -> None:
    bars = _bars()
    if early:
        bars = bars.with_columns(pl.col("available_time") - timedelta(seconds=1))
    result = build_features(bars, AppConfig(), membership=_later_membership(bars))
    assert result.height == 3
    assert result["ret_1"][0] == pytest.approx(1.0)
    assert result["decision_time"].to_list() == bars["event_time"].to_list()[1:]
    assert result.filter(pl.col("max_source_available_time") > pl.col("decision_time")).is_empty()
