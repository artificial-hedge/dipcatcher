"""Fail-closed calendar bounds: naive instants, swapped ranges, blank names.

Research/infrastructure only.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from quant_fund.calendars import CRYPTO, FX, XNYS, get_calendar, session_grid
from quant_fund.calendars.sessions import require_aware

ET = ZoneInfo("America/New_York")


def test_contains_refuses_naive_and_accepts_aware() -> None:
    session = XNYS.session_on(date(2024, 3, 11))
    assert session is not None
    with pytest.raises(ValueError, match="timezone-aware"):
        session.contains(datetime(2024, 3, 11, 14, 0))
    with pytest.raises(TypeError, match="datetime"):
        session.contains("2024-03-11")  # type: ignore[arg-type]
    assert session.contains(session.open_utc)
    assert session.contains(session.open_utc.astimezone(ET))
    assert not session.contains(session.close_utc)
    assert require_aware(session.open_utc) == session.open_utc


def test_ordered_range_still_returns_sessions() -> None:
    day = date(2024, 3, 11)
    assert len(XNYS.sessions(day, day)) == 1
    assert XNYS.sessions(date(2024, 3, 9), date(2024, 3, 9)) == []
    assert len(CRYPTO.sessions(day, day)) == 1


@pytest.mark.parametrize("calendar", [XNYS, FX, CRYPTO])
def test_inverted_session_range_fails_closed(calendar: object) -> None:
    start, end = date(2024, 3, 12), date(2024, 3, 11)
    with pytest.raises(ValueError, match="before start"):
        calendar.sessions(start, end)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="before start"):
        session_grid(calendar, start, end, timedelta(minutes=30))  # type: ignore[arg-type]


def test_timestamp_is_not_a_session_label() -> None:
    stamp = datetime(2024, 3, 11, tzinfo=UTC)
    with pytest.raises(TypeError, match="start must be a date"):
        session_grid(XNYS, stamp, date(2024, 3, 11), timedelta(minutes=30))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="end must be a date"):
        XNYS.sessions(date(2024, 3, 11), stamp)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "   ", None, 1])
def test_calendar_name_must_be_a_non_empty_string(name: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        get_calendar(name)  # type: ignore[arg-type]


def test_padded_calendar_alias_still_resolves() -> None:
    assert get_calendar("  nyse  ") is XNYS
    assert get_calendar(" forex ") is FX
