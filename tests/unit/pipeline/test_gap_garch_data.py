"""Unit tests for ``quant_fund.pipeline.train.garch_data``.

These helpers build the point-in-time return and realised-volatility histories
that feed the GARCH fit. The properties that matter are all about *not
leaking* and *not silently mis-aggregating*:

* a row is only observable once it is **published** (``available_time``), not
  merely once it happened (``event_time``);
* a null ``available_time`` on an otherwise usable row fails closed — an
  unpublished restatement is not observable evidence;
* per-name histories never fall back to a pooled cross-section;
* duplicate ``(security_id, event_time)`` keys are refused, because an
  equal-weight date mean over a double-counted name is undefined.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from quant_fund.pipeline.train.garch_data import (
    _garch_name_return_history,
    _garch_return_history,
    _realized_garch_history,
    _realized_garch_ohlc_columns,
    _require_garch_security_keys,
)
from quant_fund.schemas.errors import PointInTimeError

DAY1 = datetime(2024, 1, 2)
DAY2 = datetime(2024, 1, 3)
DAY3 = datetime(2024, 1, 4)


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _panel() -> pl.DataFrame:
    return _frame(
        [
            {
                "security_id": "AAA",
                "event_time": DAY1,
                "available_time": DAY1,
                "ret_1": 0.01,
                "high": 10.5,
                "low": 9.5,
            },
            {
                "security_id": "BBB",
                "event_time": DAY1,
                "available_time": DAY1,
                "ret_1": -0.01,
                "high": 20.5,
                "low": 19.5,
            },
            {
                "security_id": "AAA",
                "event_time": DAY2,
                "available_time": DAY2,
                "ret_1": 0.02,
                "high": 11.5,
                "low": 10.5,
            },
        ]
    )


# --------------------------------------------------------------------------- #
# Key hygiene
# --------------------------------------------------------------------------- #


def test_duplicate_security_and_time_keys_fail_closed() -> None:
    frame = _frame(
        [
            {"security_id": "AAA", "event_time": DAY1, "ret_1": 0.01},
            {"security_id": "AAA", "event_time": DAY1, "ret_1": 0.02},
        ]
    )
    with pytest.raises(PointInTimeError, match="duplicate security_id/event_time"):
        _require_garch_security_keys(frame)


def test_a_blank_security_id_fails_closed() -> None:
    frame = _frame(
        [
            {"security_id": "AAA", "event_time": DAY1, "ret_1": 0.01},
            {"security_id": "  ", "event_time": DAY1, "ret_1": 0.02},
        ]
    )
    with pytest.raises(PointInTimeError, match="blank security_id"):
        _require_garch_security_keys(frame)


def test_a_frame_without_security_keys_keeps_the_legacy_date_only_path() -> None:
    frame = _frame([{"event_time": DAY1, "ret_1": 0.01}])
    _require_garch_security_keys(frame)  # no raise


# --------------------------------------------------------------------------- #
# Date-level return history
# --------------------------------------------------------------------------- #


def test_returns_are_equal_weighted_by_date() -> None:
    dates, values = _garch_return_history(_panel())
    assert list(dates) == [DAY1, DAY2]
    # DAY1 holds +0.01 (AAA) and -0.01 (BBB) -> mean 0.0.
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.02)


def test_non_finite_returns_are_dropped() -> None:
    frame = _frame(
        [
            {"event_time": DAY1, "ret_1": 0.01},
            {"event_time": DAY2, "ret_1": float("nan")},
            {"event_time": DAY3, "ret_1": 0.03},
        ]
    )
    dates, values = _garch_return_history(frame)
    assert len(dates) == len(values) == 2
    assert all(value == value for value in values)


def test_a_frame_with_no_finite_returns_is_refused() -> None:
    frame = _frame([{"event_time": DAY1, "ret_1": float("nan")}])
    with pytest.raises(ValueError, match="no finite ret_1"):
        _garch_return_history(frame)


def test_a_frame_without_event_time_or_ret_1_is_refused() -> None:
    with pytest.raises(ValueError, match="event_time and ret_1"):
        _garch_return_history(_frame([{"event_time": DAY1}]))


def test_asof_excludes_rows_at_or_after_the_decision_origin() -> None:
    dates, _ = _garch_return_history(_panel(), asof=DAY2)
    assert list(dates) == [DAY1]


def test_asof_with_nothing_observable_returns_an_empty_history() -> None:
    dates, values = _garch_return_history(_panel(), asof=DAY1)
    assert dates.size == 0
    assert values.size == 0


def test_a_null_available_time_fails_closed_under_asof() -> None:
    """An unpublished row is not observable evidence — refuse rather than use it."""
    frame = _frame(
        [
            {"event_time": DAY1, "available_time": DAY1, "ret_1": 0.01},
            {"event_time": DAY2, "available_time": None, "ret_1": 0.02},
        ]
    )
    with pytest.raises(PointInTimeError, match="null available_time"):
        _garch_return_history(frame, asof=DAY3)


def test_a_row_published_after_the_origin_is_excluded() -> None:
    frame = _frame(
        [
            {"event_time": DAY1, "available_time": DAY1, "ret_1": 0.01},
            {"event_time": DAY2, "available_time": DAY3, "ret_1": 0.02},
        ]
    )
    dates, values = _garch_return_history(frame, asof=DAY2)
    assert list(dates) == [DAY1]
    assert values[0] == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# Per-name history
# --------------------------------------------------------------------------- #


def test_a_name_history_is_never_pooled_across_names() -> None:
    dates, values = _garch_name_return_history(_panel(), "AAA")
    assert list(dates) == [DAY1, DAY2]
    assert values[0] == pytest.approx(0.01)
    assert values[1] == pytest.approx(0.02)


def test_an_unknown_name_yields_an_empty_history() -> None:
    dates, values = _garch_name_return_history(_panel(), "ZZZ")
    assert dates.size == 0
    assert values.size == 0


def test_a_blank_name_is_refused() -> None:
    for bad in ("", "   ", None, 5):
        with pytest.raises(ValueError, match="non-empty security_id"):
            _garch_name_return_history(_panel(), bad)  # type: ignore[arg-type]


def test_a_frame_without_a_security_id_column_is_refused_for_per_name_paths() -> None:
    frame = _frame([{"event_time": DAY1, "ret_1": 0.01}])
    with pytest.raises(PointInTimeError, match="missing security_id"):
        _garch_name_return_history(frame, "AAA")


# --------------------------------------------------------------------------- #
# Realised GARCH (Parkinson)
# --------------------------------------------------------------------------- #


def test_split_adjusted_ohlc_is_preferred_over_raw_ohlc() -> None:
    frame = _frame(
        [
            {
                "event_time": DAY1,
                "ret_1": 0.01,
                "high": 10.0,
                "low": 9.0,
                "high_split_adjusted": 20.0,
                "low_split_adjusted": 18.0,
            }
        ]
    )
    assert _realized_garch_ohlc_columns(frame) == ("high_split_adjusted", "low_split_adjusted")


def test_raw_ohlc_is_used_when_no_split_adjusted_columns_exist() -> None:
    frame = _frame([{"event_time": DAY1, "ret_1": 0.01, "high": 10.0, "low": 9.0}])
    assert _realized_garch_ohlc_columns(frame) == ("high", "low")


def test_a_frame_without_ohlc_is_refused_rather_than_proxied() -> None:
    """A close-to-close squared proxy is never invented for realised variance."""
    frame = _frame([{"event_time": DAY1, "ret_1": 0.01, "close": 10.0}])
    with pytest.raises(PointInTimeError, match="refusing close-to-close squared proxy"):
        _realized_garch_ohlc_columns(frame)


def test_realized_history_pairs_returns_with_parkinson_variance() -> None:
    dates, values, measures = _realized_garch_history(_panel())
    assert list(dates) == [DAY1, DAY2]
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.02)
    assert len(measures) == 2
    assert all(measure > 0.0 for measure in measures)


def test_realized_history_drops_rows_without_a_usable_parkinson_measure() -> None:
    frame = _frame(
        [
            {"event_time": DAY1, "ret_1": 0.01, "high": 10.5, "low": 9.5},
            {"event_time": DAY2, "ret_1": 0.02, "high": 10.5, "low": 10.5},
        ]
    )
    dates, _values, _measures = _realized_garch_history(frame)
    # A zero Parkinson variance is not a usable realised measure.
    assert list(dates) == [DAY1]


def test_realized_history_requires_event_time_and_ret_1() -> None:
    frame = _frame([{"event_time": DAY1, "high": 10.5, "low": 9.5}])
    with pytest.raises(ValueError, match="event_time and ret_1"):
        _realized_garch_history(frame)


def test_realized_history_with_no_usable_pairs_is_refused() -> None:
    frame = _frame([{"event_time": DAY1, "ret_1": 0.01, "high": 10.5, "low": 10.5}])
    with pytest.raises(ValueError, match="no finite Parkinson/return pairs"):
        _realized_garch_history(frame)


def test_realized_history_under_asof_drops_unpublished_rows() -> None:
    dates, _values, _measures = _realized_garch_history(_panel(), asof=DAY2)
    assert list(dates) == [DAY1]
