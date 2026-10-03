"""SYNTHETIC funding publication lags: correctness, not market evidence."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quant_fund.backtest.sleeves import (
    _join_available_funding,
    basis_carry_hysteresis_weights,
    basis_carry_weights,
    funding_carry_weights,
    funding_spike_fade_weights,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _bars(n: int = 20) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"security_id": sid, "event_time": T0 + timedelta(hours=i), "close": 100 + np.sin(i)}
            for sid in ("A", "B")
            for i in range(n)
        ]
    )


def _funding(rows: list[tuple[str, int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "security_id": sid,
                "event_time": T0 + timedelta(hours=event),
                "available_time": T0 + timedelta(hours=available),
                "value": value,
            }
            for sid, event, available, value in rows
        ]
    )


def test_basis_waits_until_exact_availability() -> None:
    bars = _bars(6).filter(pl.col("security_id") == "A")
    funding = _funding([("A", 1, 4, 0.01)])
    weights = basis_carry_weights(bars, funding)
    assert weights["event_time"].to_list() == [T0 + timedelta(hours=h) for h in (4, 5)]
    assert weights["target_weight"].to_list() == [0.15, 0.15]


def test_delayed_new_event_keeps_prior_available_history() -> None:
    funding = _funding([("A", 0, 0, 0.02), ("A", 1, 4, -0.10)])
    joined = _join_available_funding(_bars(6), funding, 2).filter(pl.col("security_id") == "A")
    assert joined["rate_ma"].to_list() == pytest.approx([0.02] * 4 + [-0.04] * 2)


def test_late_old_print_never_replaces_newest_events() -> None:
    funding = _funding([("A", 0, 5, 0.90), ("A", 1, 1, 0.02), ("A", 2, 2, 0.04)])
    joined = _join_available_funding(_bars(7), funding, 2).filter(pl.col("security_id") == "A")
    assert joined["rate_ma"].to_list() == [None, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03]


def test_late_old_print_can_enter_unfilled_event_window() -> None:
    funding = _funding([("A", 0, 5, 0.06), ("A", 1, 1, 0.02), ("A", 2, 2, 0.04)])
    joined = _join_available_funding(_bars(7), funding, 3).filter(pl.col("security_id") == "A")
    assert joined["rate_ma"].to_list()[5:] == pytest.approx([0.04, 0.04])
    # The newest event is still .04, not the last-arriving .06.
    assert joined["_z"].to_list()[5:] == pytest.approx([0.0, 0.0], abs=1e-14)


def test_future_event_is_not_realized_even_if_available_early() -> None:
    joined = _join_available_funding(_bars(6), _funding([("A", 4, 1, 0.02)]), 2)
    assert (
        joined.filter(
            (pl.col("security_id") == "A") & (pl.col("event_time") < T0 + timedelta(hours=4))
        )["rate_ma"].null_count()
        == 4
    )


def test_shared_release_timestamp_includes_all_events() -> None:
    funding = _funding([("A", 0, 3, 0.02), ("A", 1, 3, 0.04), ("A", 2, 3, 0.06)])
    joined = _join_available_funding(_bars(5), funding.reverse(), 2).filter(
        pl.col("security_id") == "A"
    )
    assert joined["rate_ma"].to_list() == [None, None, None, 0.05, 0.05]


@pytest.mark.parametrize(
    "sleeve,kwargs",
    [
        (basis_carry_weights, {}),
        (basis_carry_hysteresis_weights, {"enter_rate": 0.0001}),
        (funding_carry_weights, {"vol_window": 4}),
        (funding_spike_fade_weights, {"vol_window": 4, "lookback_events": 12, "z_threshold": 0.1}),
    ],
)
def test_all_sleeves_ignore_unpublished_observations(sleeve, kwargs) -> None:
    bars = _bars(20)
    known = _funding(
        [
            (sid, i, i, (0.01 if sid == "A" else -0.01) * (1 + i / 10))
            for sid in ("A", "B")
            for i in range(12)
        ]
    )
    # Event is in range but publication is outside all decision bars.
    delayed = _funding([("A", 12, 24, -100.0), ("B", 12, 24, 100.0)])
    assert_frame_equal(
        sleeve(bars, known, **kwargs), sleeve(bars, pl.concat([known, delayed]), **kwargs)
    )


def test_hysteresis_enters_only_when_publication_arrives() -> None:
    weights = basis_carry_hysteresis_weights(_bars(6), _funding([("A", 1, 4, 0.01)]))
    assert weights["event_time"].to_list() == [T0 + timedelta(hours=4)]


def test_event_only_inputs_equal_explicit_immediate_availability() -> None:
    funding = _funding([("A", i, i, 0.01 * i) for i in range(10)])
    assert_frame_equal(
        _join_available_funding(_bars(), funding, 3),
        _join_available_funding(_bars(), funding.drop("available_time"), 3),
    )


@pytest.mark.parametrize("column", ["event_time", "available_time"])
def test_explicit_unknown_timestamp_fails_closed(column: str) -> None:
    funding = _funding([("A", 1, 2, 0.01)]).with_columns(
        pl.lit(None).cast(pl.Datetime("us", "UTC")).alias(column)
    )
    with pytest.raises(ValueError, match=column):
        basis_carry_weights(_bars(), funding)


def test_duplicate_observations_fail_closed() -> None:
    funding = _funding([("A", 1, 1, 0.01), ("A", 1, 4, 0.02)])
    with pytest.raises(ValueError, match="duplicate"):
        basis_carry_weights(_bars(), funding)


def test_nanosecond_availability_is_not_rounded_early() -> None:
    bars = (
        _bars(2)
        .filter(pl.col("security_id") == "A")
        .with_columns(pl.col("event_time").dt.cast_time_unit("ns"))
    )
    funding = _funding([("A", 0, 0, 0.01)]).with_columns(
        (pl.col("available_time").dt.cast_time_unit("ns").cast(pl.Int64) + 1)
        .cast(pl.Datetime("ns", "UTC"))
        .alias("available_time")
    )
    weights = basis_carry_weights(bars, funding)
    assert weights["event_time"].to_list() == [T0 + timedelta(hours=1)]


def test_randomized_release_order_matches_point_in_time_oracle() -> None:
    rng = np.random.default_rng(13)
    funding = _funding(
        [
            (sid, i, i + int(rng.integers(0, 8)), float(rng.normal()))
            for sid in ("A", "B")
            for i in range(15)
        ]
    )
    grid = _bars(25)
    actual = _join_available_funding(grid.reverse(), funding.reverse(), 5)
    for row in actual.iter_rows(named=True):
        eligible = (
            funding.filter(
                (pl.col("security_id") == row["security_id"])
                & (pl.col("event_time") <= row["event_time"])
                & (pl.col("available_time") <= row["event_time"])
            )
            .sort("event_time")
            .tail(5)
        )
        values = eligible["value"].to_numpy()
        expected = float(values.mean()) if len(values) else None
        assert (
            row["rate_ma"] == pytest.approx(expected)
            if expected is not None
            else row["rate_ma"] is None
        )
        if len(values) > 1:
            assert row["_z"] == pytest.approx(
                (values[-1] - values.mean()) / (values.std(ddof=1) + 1e-8)
            )


def test_spike_statistics_require_ten_available_events() -> None:
    funding = _funding([("A", i, i if i < 9 else 15, float(i)) for i in range(10)])
    joined = _join_available_funding(_bars(), funding, 12, min_samples=10).filter(
        pl.col("security_id") == "A"
    )
    assert joined.filter(pl.col("event_time") < T0 + timedelta(hours=15))["_z"].null_count() == 15
    assert joined.filter(pl.col("event_time") == T0 + timedelta(hours=15))["_z"][0] is not None


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_invalid_funding_rates_fail_closed(value: float | None) -> None:
    funding = _funding([("A", 1, 2, 0.01)]).with_columns(
        pl.lit(value).cast(pl.Float64).alias("value")
    )
    with pytest.raises(ValueError, match="value must be finite"):
        basis_carry_weights(_bars(), funding)
