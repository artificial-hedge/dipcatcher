"""Deterministic, bounded parser properties for Stooq daily CSV ingestion."""

from __future__ import annotations

import csv
import io
import math
from datetime import date

import polars as pl
from hypothesis import example, given, settings
from hypothesis import strategies as st

from quant_fund.data.adapters import stooq

_PRICE = st.floats(min_value=1e-6, max_value=1e9, allow_nan=False, allow_infinity=False)
_VOLUME = st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False)
_VALID_ROW = st.tuples(
    st.dates(min_value=date(1900, 1, 1), max_value=date(9999, 12, 31)),
    _PRICE,
    _PRICE,
    _PRICE,
    _PRICE,
    _VOLUME,
)
_MAX_DATE_ROW = (date(9999, 12, 31), 100.0, 102.0, 99.0, 101.0, 1000.0)
_INVALID_ROWS = (
    ("not-a-date", "1", "2", "0.5", "1.5", "10"),
    ("2024-02-30", "1", "2", "0.5", "1.5", "10"),
    ("", "1", "2", "0.5", "1.5", "10"),
    ("2024-01-03", "0", "2", "0.5", "1.5", "10"),
    ("2024-01-03", "-1", "2", "0.5", "1.5", "10"),
    ("2024-01-03", "nan", "2", "0.5", "1.5", "10"),
    ("2024-01-03", "1", "inf", "0.5", "1.5", "10"),
    ("2024-01-03", "1", "2", "-inf", "1.5", "10"),
    ("2024-01-03", "1", "2", "0.5", "NaN", "10"),
    ("2024-01-03", "1", "2", "0.5", "1.5", "-1"),
    ("2024-01-03", "1", "2", "0.5", "1.5", "nan"),
    ("2024-01-03", "1", "2", "0.5", "1.5", "inf"),
    ("2024-01-03", "not-a-number", "2", "0.5", "1.5", "10"),
    ("2024-01-03", "1", "2", "0.5"),
)
_SYMBOLS = st.sampled_from(("spy.us", "vod.uk"))


def _csv(rows: list[tuple[object, ...]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("Date", "Open", "High", "Low", "Close", "Volume"))
    writer.writerows(rows)
    return output.getvalue()


def _assert_normalized(
    frame: pl.DataFrame,
    rows: list[tuple[object, ...]],
    symbol: str,
) -> None:
    assert frame.height == len(rows)
    for actual, expected in zip(frame.iter_rows(named=True), rows, strict=True):
        day, opn, high, low, close, volume = expected
        assert isinstance(day, date)
        prices = (opn, high, low, close)
        assert all(isinstance(value, float) for value in (*prices, volume))
        assert actual["security_id"] == "TEST"
        assert actual["symbol"] == "TEST"
        assert actual["open"] == opn
        assert actual["high"] == max(prices)
        assert actual["low"] == min(prices)
        assert actual["close"] == close
        assert actual["volume"] == volume
        assert actual["event_time"] == stooq.session_close(day, symbol)
        assert actual["available_time"] == actual["event_time"]
        assert actual["source"] == stooq.SOURCE
        assert actual["revision_id"] == stooq.REVISION
        assert actual["currency"] == ("GBP" if symbol.endswith(".uk") else "USD")
        assert actual["session"] == "rth"


def _assert_output_invariants(frame: pl.DataFrame) -> None:
    for row in frame.iter_rows(named=True):
        prices = tuple(float(row[name]) for name in ("open", "high", "low", "close"))
        volume = float(row["volume"])
        assert all(math.isfinite(value) and value > 0 for value in prices)
        assert math.isfinite(volume) and volume >= 0
        assert row["high"] == max(prices)
        assert row["low"] == min(prices)
        assert row["event_time"] == row["available_time"]


@example(
    rows=[(date(2024, 3, 11), 100.0, 99.0, 102.0, 101.0, 1000.0)],
    symbol="spy.us",
)
@example(rows=[_MAX_DATE_ROW], symbol="spy.us")
@given(rows=st.lists(_VALID_ROW, max_size=8), symbol=_SYMBOLS)
@settings(max_examples=40, deadline=None, derandomize=True)
def test_valid_csv_round_trips_and_normalizes(rows: list[tuple[object, ...]], symbol: str) -> None:
    frame = stooq.parse_stooq_csv(_csv(rows), security_id="TEST", stooq_symbol=symbol)
    _assert_normalized(frame, rows, symbol)


@given(
    valid_rows=st.lists(_VALID_ROW, max_size=5),
    invalid_rows=st.lists(st.sampled_from(_INVALID_ROWS), min_size=1, max_size=6),
    symbol=_SYMBOLS,
)
@settings(max_examples=50, deadline=None, derandomize=True)
def test_malformed_rows_are_skipped_without_poisoning_valid_rows(
    valid_rows: list[tuple[object, ...]],
    invalid_rows: list[tuple[object, ...]],
    symbol: str,
) -> None:
    rows = valid_rows[:1] + invalid_rows + valid_rows[1:]
    frame = stooq.parse_stooq_csv(_csv(rows), security_id="TEST", stooq_symbol=symbol)
    _assert_normalized(frame, valid_rows, symbol)


@example(text="\r0")
@given(
    text=st.text(
        alphabet=st.characters(min_codepoint=0, max_codepoint=127),
        max_size=384,
    )
)
@settings(max_examples=60, deadline=None, derandomize=True)
def test_arbitrary_ascii_csv_never_emits_invalid_bars(text: str) -> None:
    frame = stooq.parse_stooq_csv(text, security_id="TEST", stooq_symbol="spy.us")
    _assert_output_invariants(frame)
