"""Hypothesis invariants for the trading-calendar layer.

The core guarantee: across DST transitions and half days, every generated
bar tiles its session exactly — no misalignment, no duplicates, no gaps
within session bounds — and every in-session timestamp maps to a valid slot.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from typing import cast
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.calendars import (
    CRYPTO,
    FX,
    XNYS,
    assign_bars,
    bars_in_session,
    grid_invariants,
    session_grid,
)
from quant_fund.calendars.sessions import Session, TradingCalendar

ET = ZoneInfo("America/New_York")
RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False

# Ranges that straddle the 2024 US DST transitions plus ordinary weeks.
DST_WINDOWS = [
    (date(2024, 3, 4), date(2024, 3, 15)),  # spring-forward week (Mar 10)
    (date(2024, 10, 28), date(2024, 11, 8)),  # fall-back week (Nov 3)
    (date(2024, 6, 24), date(2024, 7, 5)),  # half day + holiday
    (date(2024, 12, 23), date(2024, 12, 31)),  # Christmas Eve/Christmas
    (date(2024, 11, 25), date(2024, 12, 2)),  # Thanksgiving + half day
    (date(2012, 10, 26), date(2012, 11, 2)),  # Sandy
    (date(2001, 9, 10), date(2001, 9, 18)),  # 9/11 closure week
]

BAR_SIZES = st.sampled_from(
    [timedelta(minutes=m) for m in (1, 2, 3, 5, 6, 7, 10, 13, 15, 30, 45)]
    + [timedelta(hours=h) for h in (1, 2, 3)]
)

CALENDARS: list[tuple[str, TradingCalendar]] = [
    ("XNYS", XNYS),
    ("CRYPTO", CRYPTO),
    ("FX", FX),
]


def _assert_tiled(session: Session, bars: list[tuple[int, datetime, datetime]]) -> None:
    """Bars must exactly partition ``[open_utc, close_utc)``."""
    if not bars:
        return
    assert bars[0][1] == session.open_utc
    assert bars[-1][2] == session.close_utc
    assert [b[0] for b in bars] == list(range(len(bars)))
    for (_, start, end), (_, next_start, _) in zip(bars[:-1], bars[1:], strict=True):
        assert start < end <= session.close_utc
        assert end == next_start  # no gaps, no overlaps


@given(
    calendar_name=st.sampled_from([name for name, _ in CALENDARS]),
    window=st.sampled_from(DST_WINDOWS),
    bar_size=BAR_SIZES,
)
@settings(max_examples=80, deadline=None)
def test_grid_tiles_every_session_without_gaps(
    calendar_name: str, window: tuple[date, date], bar_size: timedelta
) -> None:
    calendar = dict(CALENDARS)[calendar_name]
    start, end = window
    grid = session_grid(calendar, start, end, bar_size)
    assert grid_invariants(grid) == []
    # Local-label columns must be tz-aware in the exchange zone; UTC columns UTC.
    if grid.height:
        assert grid["bar_open_local"].dtype == pl.Datetime("us", calendar.tz.key)
        assert grid["bar_open_utc"].dtype == pl.Datetime("us", "UTC")


@given(
    d=st.dates(min_value=date(1998, 1, 5), max_value=date(2030, 12, 30)),
    bar_size=BAR_SIZES,
)
@settings(max_examples=80, deadline=None)
def test_nyse_session_shape_and_tiling(d: date, bar_size: timedelta) -> None:
    session = XNYS.session_on(d)
    if session is None:
        assert not XNYS.is_session_day(d)
        return
    # Local wall clock is invariant even when the UTC offset moves.
    assert session.open_utc.astimezone(ET).time() == time(9, 30)
    close_local = session.close_utc.astimezone(ET)
    assert close_local.time() in (time(13, 0), time(16, 0))
    assert close_local.date() == d
    # Session is either a 390-minute regular day or a 210-minute early close.
    assert session.duration in (timedelta(minutes=390), timedelta(minutes=210))
    assert session.is_half_day == (session.duration == timedelta(minutes=210))
    _assert_tiled(session, bars_in_session(session, bar_size))


@given(
    d=st.dates(min_value=date(1998, 1, 5), max_value=date(2030, 12, 30)),
    offset_minutes=st.integers(min_value=0, max_value=60 * 30),
    bar_size=BAR_SIZES,
)
@settings(max_examples=80, deadline=None)
def test_every_in_session_timestamp_maps_to_a_valid_slot(
    d: date, offset_minutes: int, bar_size: timedelta
) -> None:
    session = XNYS.session_on(d)
    if session is None:
        return
    span = int(session.duration.total_seconds() // 60)  # 390 or 210
    ts = session.open_utc + timedelta(minutes=offset_minutes % span)
    out = assign_bars(XNYS, [ts], bar_size)
    row = out.row(0, named=True)
    assert row["session_date"] == d
    assert row["bar_index"] is not None
    assert row["bar_open_utc"] <= ts < row["bar_close_utc"]
    # The assigned slot is exactly the floor-division bucket from the open.
    expected_index = int((ts - session.open_utc) // bar_size)
    assert row["bar_index"] == expected_index
    assert row["bar_open_utc"] == session.open_utc + expected_index * bar_size


@given(
    window=st.sampled_from(DST_WINDOWS),
    hour_offset=st.integers(min_value=0, max_value=47),
)
@settings(max_examples=60, deadline=None)
def test_out_of_session_timestamps_never_map(window: tuple[date, date], hour_offset: int) -> None:
    """Weekend/gap timestamps must produce null slots — never a phantom bar."""
    start, _ = window
    # Find a guaranteed non-session instant: Saturday noon UTC in the window.
    d = start
    while d.weekday() != 5:  # advance to a Saturday
        d += timedelta(days=1)
    ts = datetime(d.year, d.month, d.day, 12, 0, tzinfo=UTC) + timedelta(hours=hour_offset % 24)
    out = assign_bars(XNYS, [ts], timedelta(minutes=5))
    assert out["bar_index"].to_list() == [None]
    out_fx = assign_bars(FX, [ts], timedelta(minutes=5))
    # FX has no session Saturday 12:00 UTC either (market is closed).
    assert out_fx["bar_index"].to_list() == [None]


@given(
    monday=st.dates(min_value=date(1998, 1, 5), max_value=date(2030, 12, 25)).filter(
        lambda d: d.weekday() == 0
    ),
)
@settings(max_examples=40, deadline=None)
def test_fx_week_is_five_24h_sessions_with_weekend_gap(monday: date) -> None:
    sessions = FX.sessions(monday, monday + timedelta(days=6))
    assert [s.date for s in sessions] == [monday + timedelta(days=i) for i in range(5)]
    for session in sessions:
        # Exactly 24h even across the DST boundary — the offset change always
        # lands inside the weekend gap, never inside a session.
        assert session.duration == timedelta(hours=24)
        assert session.close_utc.astimezone(ET).time() == time(17, 0)
    for prev, nxt in zip(sessions[:-1], sessions[1:], strict=True):
        assert prev.close_utc == nxt.open_utc  # contiguous within the week
    # The weekend gap: Friday close to the next Monday session's open is > 0.
    assert sessions[0].open_utc.weekday() == 6  # Sunday ET open in UTC terms


@given(
    d=st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 30)),
    bar_size=BAR_SIZES,
)
@settings(max_examples=60, deadline=None)
def test_crypto_tiles_full_utc_day(d: date, bar_size: timedelta) -> None:
    session = CRYPTO.session_on(d)
    assert session.duration == timedelta(hours=24)
    _assert_tiled(session, bars_in_session(session, bar_size))


@given(
    window=st.sampled_from(DST_WINDOWS),
    bar_size=BAR_SIZES,
)
@settings(max_examples=60, deadline=None)
def test_no_duplicate_or_overlapping_bars_in_range(
    window: tuple[date, date], bar_size: timedelta
) -> None:
    start, end = window
    for _, calendar in CALENDARS:
        grid = session_grid(calendar, start, end, bar_size)
        assert grid_invariants(grid) == []
        if grid.is_empty():
            continue
        opens = grid["bar_open_utc"].to_list()
        assert len(opens) == len(set(opens))  # no duplicate bar starts
        # Bars never overlap across the whole grid (sessions are disjoint).
        ordered = grid.sort("bar_open_utc")
        starts = ordered["bar_open_utc"].to_list()
        ends = ordered["bar_close_utc"].to_list()
        for prev_end, next_start in zip(ends[:-1], starts[1:], strict=True):
            assert prev_end <= next_start


@given(
    d=st.dates(min_value=date(1998, 1, 5), max_value=date(2030, 12, 30)),
)
@settings(max_examples=60, deadline=None)
def test_session_dates_are_exchange_local_labels(d: date) -> None:
    """Session local open/close dates agree with the label for daily markets."""
    session = XNYS.session_on(d)
    if session is not None:
        assert session.open_utc.astimezone(ET).date() == d
        assert session.close_utc.astimezone(ET).date() == d


def test_grid_assignment_round_trip_consistency() -> None:
    """Concrete spot check: assigned slot == grid row for the same session."""
    grid = session_grid(XNYS, date(2024, 3, 11), date(2024, 3, 11), timedelta(minutes=5))
    ts = datetime(2024, 3, 11, 14, 37, 30, tzinfo=UTC)  # inside bar 13
    out = assign_bars(XNYS, [ts], timedelta(minutes=5))
    row = out.row(0, named=True)
    match = grid.filter(
        (pl.col("session_date") == row["session_date"]) & (pl.col("bar_index") == row["bar_index"])
    )
    assert match.height == 1
    m = match.row(0, named=True)
    assert m["bar_open_utc"] == row["bar_open_utc"]
    assert m["bar_close_utc"] == row["bar_close_utc"]
    assert math.floor((ts - m["session_open_utc"]).total_seconds() / 300) == row["bar_index"]


INVALID_RANGE_LABELS = [
    pytest.param(datetime(2024, 1, 2), id="naive-datetime"),
    pytest.param(datetime(2024, 1, 2, tzinfo=UTC), id="aware-datetime"),
    pytest.param("2024-01-02", id="string"),
]


@pytest.mark.parametrize("calendar", [XNYS, FX, CRYPTO], ids=lambda calendar: calendar.name)
def test_sessions_reject_inverted_range(calendar: TradingCalendar) -> None:
    with pytest.raises(ValueError, match="start"):
        calendar.sessions(date(2024, 1, 3), date(2024, 1, 2))


@pytest.mark.parametrize("calendar", [XNYS, FX, CRYPTO], ids=lambda calendar: calendar.name)
def test_session_grid_rejects_inverted_range(calendar: TradingCalendar) -> None:
    with pytest.raises(ValueError, match="start"):
        session_grid(calendar, date(2024, 1, 3), date(2024, 1, 2), timedelta(minutes=5))


@pytest.mark.parametrize("calendar", [XNYS, FX, CRYPTO], ids=lambda calendar: calendar.name)
@pytest.mark.parametrize("bad_label", INVALID_RANGE_LABELS)
def test_sessions_reject_non_date_range_labels(
    calendar: TradingCalendar, bad_label: object
) -> None:
    valid_label = date(2024, 1, 2)
    for start, end in ((bad_label, valid_label), (valid_label, bad_label)):
        with pytest.raises(TypeError, match="date"):
            calendar.sessions(cast(date, start), cast(date, end))


@pytest.mark.parametrize("calendar", [XNYS, FX, CRYPTO], ids=lambda calendar: calendar.name)
@pytest.mark.parametrize("bad_label", INVALID_RANGE_LABELS)
def test_session_grid_rejects_non_date_range_labels(
    calendar: TradingCalendar, bad_label: object
) -> None:
    valid_label = date(2024, 1, 2)
    for start, end in ((bad_label, valid_label), (valid_label, bad_label)):
        with pytest.raises(TypeError, match="date"):
            session_grid(calendar, cast(date, start), cast(date, end), timedelta(minutes=5))


def test_xnys_weekend_only_direct_and_grid_ranges_are_empty() -> None:
    saturday, sunday = date(2024, 1, 6), date(2024, 1, 7)
    assert XNYS.sessions(saturday, sunday) == []
    grid = session_grid(XNYS, saturday, sunday, timedelta(minutes=5))
    assert grid.is_empty()
    assert grid_invariants(grid) == []


def test_grid_invariants_rejects_overlapping_sessions_for_same_calendar() -> None:
    base = session_grid(XNYS, date(2024, 1, 2), date(2024, 1, 2), timedelta(minutes=5))
    offset = timedelta(hours=1)
    timestamp_columns = [
        "session_open_utc",
        "session_close_utc",
        "bar_open_utc",
        "bar_close_utc",
        "bar_open_local",
        "bar_close_local",
    ]
    overlapping = base.with_columns(
        pl.lit(date(2024, 1, 3)).alias("session_date"),
        *[pl.col(column) + offset for column in timestamp_columns],
    )

    errors = grid_invariants(pl.concat([base, overlapping]))

    assert any("overlap" in error.lower() for error in errors), errors


def test_grid_invariants_rejects_shared_open_with_multiple_session_dates() -> None:
    base = session_grid(XNYS, date(2024, 1, 2), date(2024, 1, 2), timedelta(minutes=5))
    other_date = base.with_columns(pl.lit(date(2024, 1, 3)).alias("session_date"))

    errors = grid_invariants(pl.concat([base, other_date]))

    assert any(
        "session_date" in error
        and "session_open_utc" in error
        and ("multiple" in error or "inconsistent" in error)
        for error in errors
    ), errors


def test_grid_invariants_rejects_inconsistent_session_close_within_identity() -> None:
    base = session_grid(XNYS, date(2024, 1, 2), date(2024, 1, 2), timedelta(minutes=5))
    inconsistent = base.with_columns(
        pl.when(pl.col("bar_index") == 0)
        .then(pl.col("session_close_utc") - timedelta(minutes=5))
        .otherwise(pl.col("session_close_utc"))
        .alias("session_close_utc")
    )

    errors = grid_invariants(inconsistent)

    assert any(
        "session_close_utc" in error and ("multiple" in error or "inconsistent" in error)
        for error in errors
    ), errors


def test_grid_invariants_allows_distinct_calendars_sharing_open() -> None:
    base = session_grid(XNYS, date(2024, 1, 2), date(2024, 1, 2), timedelta(minutes=5))
    other_calendar = base.with_columns(pl.lit("OTHER").alias("calendar"))

    assert grid_invariants(pl.concat([base, other_calendar])) == []


def test_grid_invariants_allows_contiguous_fx_sessions() -> None:
    grid = session_grid(FX, date(2024, 1, 2), date(2024, 1, 3), timedelta(hours=1))
    bounds = (
        grid.select("session_date", "session_open_utc", "session_close_utc")
        .unique()
        .sort("session_open_utc")
    )
    assert bounds.height == 2
    assert bounds["session_close_utc"][0] == bounds["session_open_utc"][1]
    assert grid_invariants(grid) == []
