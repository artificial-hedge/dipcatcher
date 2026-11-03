"""Unit tests for ``quant_fund.calendars.us_equities``.

The NYSE/Nasdaq calendar is the reference session clock for every US-equity
research path, so two properties are load-bearing:

* session bounds are built from **exchange-local wall times**, which means DST
  is handled structurally — 09:30 ET is 14:30 UTC under EST and 13:30 UTC
  under EDT without a second table of offsets;
* weekends, scheduled holidays and *unscheduled* closures (9/11, Sandy) have
  no session at all, and scheduled early closes get a shortened session.

A calendar that silently returns a session on a closure day would put a
research artifact on a day the market never traded.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from quant_fund.calendars.holidays import UNSCHEDULED_CLOSURES
from quant_fund.calendars.us_equities import (
    EARLY_CLOSE,
    REGULAR_CLOSE,
    REGULAR_OPEN,
    XNYS,
    UsEquitiesCalendar,
)

UTC = ZoneInfo("UTC")


# --------------------------------------------------------------------------- #
# Session days
# --------------------------------------------------------------------------- #


def test_weekends_have_no_session() -> None:
    assert XNYS.session_on(date(2024, 1, 6)) is None  # Saturday
    assert XNYS.session_on(date(2024, 1, 7)) is None  # Sunday
    assert not XNYS.is_session_day(date(2024, 1, 6))


def test_scheduled_holidays_have_no_session() -> None:
    for holiday in (
        date(2024, 1, 1),  # New Year's Day (observed)
        date(2024, 7, 4),  # Independence Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
    ):
        assert not XNYS.is_session_day(holiday), holiday
        assert XNYS.session_on(holiday) is None


def test_unscheduled_closures_have_no_session() -> None:
    """9/11 and Hurricane Sandy are closures even though they are not holidays."""
    for closed in (date(2001, 9, 11), date(2012, 10, 29), date(2012, 10, 30)):
        assert closed in UNSCHEDULED_CLOSURES
        assert not XNYS.is_session_day(closed), closed


def test_a_normal_weekday_is_a_session_day() -> None:
    assert XNYS.is_session_day(date(2024, 1, 2))
    session = XNYS.session_on(date(2024, 1, 2))
    assert session is not None
    assert session.date == date(2024, 1, 2)
    assert session.is_half_day is False


# --------------------------------------------------------------------------- #
# Session bounds and DST
# --------------------------------------------------------------------------- #


def test_regular_session_bounds_are_exchange_local() -> None:
    session = XNYS.session_on(date(2024, 1, 2))
    assert session is not None
    local_open = session.open_utc.astimezone(ZoneInfo("America/New_York"))
    local_close = session.close_utc.astimezone(ZoneInfo("America/New_York"))
    assert local_open.time() == REGULAR_OPEN
    assert local_close.time() == REGULAR_CLOSE


def test_session_bounds_are_dst_structural_not_offset_tables() -> None:
    """09:30 ET is 14:30 UTC under EST and 13:30 UTC under EDT."""
    before_dst = XNYS.session_on(date(2024, 3, 8))
    after_dst = XNYS.session_on(date(2024, 3, 11))
    assert before_dst is not None
    assert after_dst is not None

    assert before_dst.open_utc == datetime(2024, 3, 8, 14, 30, tzinfo=UTC)
    assert before_dst.close_utc == datetime(2024, 3, 8, 21, 0, tzinfo=UTC)
    assert after_dst.open_utc == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    assert after_dst.close_utc == datetime(2024, 3, 11, 20, 0, tzinfo=UTC)


def test_session_bounds_are_utc_aware() -> None:
    session = XNYS.session_on(date(2024, 1, 2))
    assert session is not None
    assert session.open_utc.tzinfo is not None
    assert session.close_utc.tzinfo is not None
    assert session.open_utc.utcoffset() == timedelta(0)


def test_an_early_close_shortens_the_session() -> None:
    """July 3 2024 is a 13:00 ET early close."""
    session = XNYS.session_on(date(2024, 7, 3))
    assert session is not None
    assert session.is_half_day is True
    local_close = session.close_utc.astimezone(ZoneInfo("America/New_York"))
    assert local_close.time() == EARLY_CLOSE
    # The open is unchanged: an early close shortens the tail, not the head.
    local_open = session.open_utc.astimezone(ZoneInfo("America/New_York"))
    assert local_open.time() == REGULAR_OPEN


def test_early_close_uses_the_wall_clock_not_a_fixed_offset() -> None:
    """A November early close is 18:00 UTC (EST); a July one is 17:00 UTC (EDT)."""
    july = XNYS.session_on(date(2024, 7, 3))
    november = XNYS.session_on(date(2024, 11, 29))
    assert july is not None
    assert november is not None
    assert july.close_utc == datetime(2024, 7, 3, 17, 0, tzinfo=UTC)
    assert november.close_utc == datetime(2024, 11, 29, 18, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Ranges and navigation
# --------------------------------------------------------------------------- #


def test_sessions_returns_only_trading_days_in_the_closed_range() -> None:
    sessions = XNYS.sessions(date(2024, 1, 1), date(2024, 1, 7))
    assert [session.date for session in sessions] == [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
    ]


def test_sessions_are_ordered_and_cover_the_range_exactly() -> None:
    sessions = XNYS.sessions(date(2024, 6, 28), date(2024, 7, 8))
    dates = [session.date for session in sessions]
    assert dates == sorted(dates)
    assert date(2024, 7, 4) not in dates
    assert date(2024, 7, 3) in dates


def test_next_session_day_is_strictly_after_the_given_day() -> None:
    assert XNYS.next_session_day(date(2024, 1, 5)) == date(2024, 1, 8)
    # A Friday holiday rolls forward over the weekend.
    assert XNYS.next_session_day(date(2024, 6, 28)) == date(2024, 7, 1)


def test_previous_session_day_is_strictly_before_the_given_day() -> None:
    assert XNYS.previous_session_day(date(2024, 1, 8)) == date(2024, 1, 5)
    assert XNYS.previous_session_day(date(2024, 7, 1)) == date(2024, 6, 28)


def test_navigation_skips_an_unscheduled_closure() -> None:
    """Sandy closed 2012-10-29 and 2012-10-30 — the next session is the 31st."""
    assert XNYS.next_session_day(date(2012, 10, 28)) == date(2012, 10, 31)
    assert XNYS.previous_session_day(date(2012, 10, 31)) == date(2012, 10, 26)


# --------------------------------------------------------------------------- #
# Calendar identity
# --------------------------------------------------------------------------- #


def test_the_calendar_reports_the_xnys_identity() -> None:
    assert XNYS.name == "XNYS"
    assert UsEquitiesCalendar().name == "XNYS"


def test_calendar_instances_are_independent_but_agree() -> None:
    other = UsEquitiesCalendar()
    assert other.sessions(date(2024, 1, 1), date(2024, 1, 31)) == XNYS.sessions(
        date(2024, 1, 1), date(2024, 1, 31)
    )


def test_the_calendar_rejects_pre_1995_holiday_queries() -> None:
    """The holiday table only supports 1995 and later."""
    with pytest.raises(ValueError):
        XNYS.is_session_day(date(1994, 6, 15))


def test_session_bounds_never_precede_the_exchange_open() -> None:
    """Sanity: every session opens before it closes, and both are UTC-based."""
    for session in XNYS.sessions(date(2024, 1, 1), date(2024, 12, 31)):
        assert session.open_utc < session.close_utc
        assert session.open_utc.utcoffset() == timedelta(0)
        assert session.close_utc.utcoffset() == timedelta(0)
