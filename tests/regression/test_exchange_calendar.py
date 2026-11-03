"""NYSE hours, holidays, early closes, and DST on the offline live path.

SYNTHETIC. No socket and no live-P&L claim.
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.calendars.us_equities import ET, XNYS
from quant_fund.live.exchange_clock import nyse_rth_closed
from tests.unit.fault.support import FaultBroker, arm, buy, open_desk, open_fault_session

_FAST = settings(derandomize=True, max_examples=20, deadline=None)


def _ns(day: date, hour: int, minute: int, *, zone: str = "America/New_York") -> int:
    stamp = datetime(day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(zone))
    return int(stamp.timestamp() * 1_000_000_000)


def _order(tmp: Path, stamp: int, name: str) -> tuple[dict[str, object], FaultBroker]:
    broker = FaultBroker()
    session = open_fault_session(tmp / name, broker, now=stamp)
    arm(session)
    return buy(session, client_order_id=name, at_ns=stamp, mark=100.0, limit_price=100.0), broker


@given(st.dates(min_value=date(2024, 1, 1), max_value=date(2026, 12, 31)))
@_FAST
def test_ten_thirty_follows_the_nyse_session_day(day: date) -> None:
    """10:30 America/New_York is tradable only on an NYSE session day."""
    stamp = _ns(day, 10, 30)
    with TemporaryDirectory() as raw:
        report, broker = _order(Path(raw), stamp, "d")
    if XNYS.is_session_day(day):
        assert report["ok"] is True
        assert broker.submit_count == 1
    else:
        assert report["broker_called"] is False
        assert report["reasons"] == ["market_closed"]
        assert broker.submit_count == 0


def test_holidays_early_closes_and_dst_transitions(tmp_path: Path) -> None:
    holidays = (
        date(2024, 1, 1),  # New Year
        date(2024, 1, 15),  # MLK Day
        date(2024, 7, 4),
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),
        date(2025, 1, 9),  # Carter day of mourning
    )
    for day in holidays:
        assert XNYS.is_session_day(day) is False
        stamp = _ns(day, 10, 0)
        assert nyse_rth_closed(stamp) is True
        report, broker = _order(tmp_path, stamp, day.isoformat())
        assert report["reasons"] == ["market_closed"]
        assert broker.submit_count == 0

    early = date(2024, 11, 29)
    session = XNYS.session_on(early)
    assert session is not None and session.is_half_day
    assert session.close_utc.astimezone(ET).time() == time(13, 0)
    before, before_broker = _order(tmp_path, _ns(early, 12, 59), "early-open")
    assert before["ok"] is True
    assert before_broker.submit_count == 1
    after, after_broker = _order(tmp_path, _ns(early, 13, 0), "early-close")
    assert after["reasons"] == ["market_closed"]
    assert after_broker.submit_count == 0

    july3 = date(2024, 7, 3)
    assert XNYS.session_on(july3) is not None
    assert XNYS.session_on(july3).is_half_day  # type: ignore[union-attr]
    july_close, july_broker = _order(tmp_path, _ns(july3, 13, 0), "july3")
    assert july_close["reasons"] == ["market_closed"]
    assert july_broker.submit_count == 0

    # Spring forward is 2024-03-10. Monday's 09:30 is 13:30 UTC.
    spring = date(2024, 3, 11)
    assert nyse_rth_closed(_ns(spring, 9, 29)) is True
    assert nyse_rth_closed(_ns(spring, 9, 30)) is False
    spring_utc = int(
        datetime(2024, 3, 11, 14, 0, tzinfo=ZoneInfo("UTC")).timestamp() * 1_000_000_000
    )
    assert datetime.fromtimestamp(spring_utc / 1e9, ET).hour == 10
    spring_ok, spring_broker = _order(tmp_path, spring_utc, "spring-utc")
    assert spring_ok["ok"] is True
    assert spring_broker.submit_count == 1

    # Fall back is 2024-11-03. Monday's 09:30 is 14:30 UTC, so 14:00 UTC is still closed.
    fall = date(2024, 11, 4)
    assert nyse_rth_closed(_ns(fall, 9, 30)) is False
    fall_utc = int(datetime(2024, 11, 4, 14, 0, tzinfo=ZoneInfo("UTC")).timestamp() * 1_000_000_000)
    assert datetime.fromtimestamp(fall_utc / 1e9, ET).hour == 9
    fall_closed, fall_broker = _order(tmp_path, fall_utc, "fall-utc")
    assert fall_closed["reasons"] == ["market_closed"]
    assert fall_broker.submit_count == 0


def test_desk_holiday_does_not_transmit(tmp_path: Path) -> None:
    desk = open_desk(tmp_path / "desk")
    try:
        opened = datetime(2024, 1, 3, 15, 0, tzinfo=ET)
        desk.arm("loopback", "", int(opened.timestamp() * 1_000_000_000))
        holiday = _ns(date(2024, 11, 28), 10, 30)
        denied = desk.submit("AAPL", "BUY", 10, 100.0, holiday, client_order_id="holiday")
        assert denied["accepted"] is False
        assert denied["transmitted"] is False
        assert denied["order"]["reasons"] == ["market_closed"]  # type: ignore[index]
        assert desk._loopback._queued == []
    finally:
        desk.close()
