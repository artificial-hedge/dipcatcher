"""Adversarial live risk limits. SYNTHETIC gateway checks, not market evidence.

Splitting, rapid-fire, partial fills, and concurrent submits must not push
aggregate exposure through a ceiling. Loss and drawdown latch on marks and
fills, survive restart and the New York day boundary, and cancel working
orders. Shorts, halts, off-hours, and off-tick orders die before the broker.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_fund.live.brokers import LiveOrder
from quant_fund.live.errors import BrokerRejected
from quant_fund.pretrade.config import LimitConfig
from tests.unit.fault.support import (
    TS,
    FaultBroker,
    arm,
    buy,
    open_desk,
    open_fault_session,
)

_STEP = 2_100_000_000
_ET = ZoneInfo("America/New_York")


def _ns(year: int, month: int, day: int, hour: int, minute: int) -> int:
    stamp = datetime(year, month, day, hour, minute, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _limits(session: object, **updates: object) -> LimitConfig:
    return session.pretrade.limits.model_copy(update=updates)  # type: ignore[attr-defined, no-any-return]


class _Cancels(FaultBroker):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls = 0

    def cancel(self, client_order_id: str, broker_order_id: str):  # type: ignore[no-untyped-def]
        self.cancel_calls += 1
        return super().cancel(client_order_id, broker_order_id)


class _Fills(FaultBroker):
    """Paper venue that prints the whole order on submit. Still no socket."""

    def __init__(self) -> None:
        super().__init__()
        self.fill_on_submit = True

    def submit(self, order: LiveOrder):  # type: ignore[no-untyped-def]
        ack = super().submit(order)
        if not self.fill_on_submit or ack.status != "accepted":
            return ack
        self.deliver(
            order.client_order_id,
            order.quantity,
            order.limit_price,
            order.fill_time_ns,
            f"fill-{order.client_order_id}",
        )
        return self._acks[order.client_order_id]


class _RejectsCancel(_Cancels):
    def cancel(self, client_order_id: str, broker_order_id: str):  # type: ignore[no-untyped-def]
        self.cancel_calls += 1
        raise BrokerRejected("cancel refused")


class _Overlap:
    """Counts how many submits are inside the broker at once."""

    def __init__(self, inner: FaultBroker) -> None:
        self.inner = inner
        self.max_inside = 0
        self._inside = 0
        self._lock = threading.Lock()
        self.submit_count = 0

    def submit(self, order: LiveOrder):  # type: ignore[no-untyped-def]
        with self._lock:
            self._inside += 1
            self.max_inside = max(self.max_inside, self._inside)
            self.submit_count += 1
        try:
            time.sleep(0.02)
            return self.inner.submit(order)
        finally:
            with self._lock:
                self._inside -= 1

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def test_split_buys_stop_at_the_position_quantity(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    stamp = TS
    for index in (1, 2):
        result = buy(
            session,
            quantity=8_000.0,
            limit_price=10.0,
            mark=10.0,
            client_order_id=f"split-{index}",
            at_ns=stamp,
        )
        assert result["ok"] is True
        assert result["broker_called"] is True
        stamp += _STEP
    denied = buy(
        session,
        quantity=8_000.0,
        limit_price=10.0,
        mark=10.0,
        client_order_id="split-3",
        at_ns=stamp,
    )
    assert denied["ok"] is False
    assert denied["broker_called"] is False
    assert "position_quantity" in denied["reasons"]
    assert broker.submit_count == 2


def test_rapid_fire_distinct_sizes_stop_at_the_position_cap(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    accepted = 0.0
    accepted_orders = 0
    for index, qty in enumerate(range(1_000, 1_030)):
        result = buy(
            session,
            quantity=float(qty),
            limit_price=10.0,
            mark=10.0,
            client_order_id=f"rapid-{index}",
            at_ns=TS + index,
        )
        if result["ok"]:
            accepted += qty
            accepted_orders += 1
            assert result["broker_called"] is True
        else:
            assert result["broker_called"] is False
    assert accepted <= 20_000.0
    assert broker.submit_count == accepted_orders
    reserved, _cash = session._working_exposure()
    assert reserved.get("AAPL", 0.0) == accepted


def test_fat_finger_quantity_and_notional_never_reach_the_broker(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    qty = buy(session, quantity=10_001.0, limit_price=1.0, mark=1.0, client_order_id="fat-qty")
    assert qty["broker_called"] is False
    assert "max_order_quantity" in qty["reasons"]
    notion = buy(
        session,
        quantity=2_600.0,
        limit_price=100.0,
        mark=100.0,
        client_order_id="fat-notion",
        at_ns=TS + 1,
    )
    assert notion["broker_called"] is False
    assert "max_order_notional" in notion["reasons"]
    assert broker.submit_count == 0


def test_tightened_gross_and_net_bind_before_buying_power(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    gross = session.update_limits(_limits(session, max_gross_notional=150_000.0), actor="ada")
    assert gross["applied"] is True
    assert buy(session, quantity=1_000.0, client_order_id="gross-1")["ok"] is True
    denied = buy(
        session,
        symbol="MSFT",
        quantity=1_000.0,
        client_order_id="gross-2",
        at_ns=TS + 1,
    )
    assert denied["broker_called"] is False
    assert "gross_exposure" in denied["reasons"]
    assert "buying_power" not in denied["reasons"]

    session.update_limits(
        _limits(session, max_gross_notional=2_000_000.0, max_net_notional=150_000.0),
        actor="ada",
        reauthorize=True,
    )
    net = buy(
        session,
        symbol="GOOG",
        quantity=1_000.0,
        client_order_id="net-1",
        at_ns=TS + 2,
    )
    assert net["broker_called"] is False
    assert "net_exposure" in net["reasons"]
    assert "buying_power" not in net["reasons"]


def test_partial_fill_counts_the_remainder_once(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    opened = buy(
        session,
        quantity=10_000.0,
        limit_price=10.0,
        mark=10.0,
        client_order_id="part-1",
    )
    assert opened["status"] == "accepted"
    assert broker.deliver("part-1", 4_000.0, 10.0, TS, "part-fill") == "applied"
    topped = buy(
        session,
        quantity=10_000.0,
        limit_price=10.0,
        mark=10.0,
        client_order_id="part-2",
        at_ns=TS + _STEP,
    )
    assert topped["ok"] is True
    overflow = buy(
        session,
        quantity=1.0,
        limit_price=10.0,
        mark=10.0,
        client_order_id="part-3",
        at_ns=TS + 2 * _STEP,
    )
    assert overflow["ok"] is False
    assert overflow["broker_called"] is False
    assert "position_quantity" in overflow["reasons"]
    reserved, _cash = session._working_exposure()
    held = float(broker.positions.get("AAPL", 0.0)) + float(reserved.get("AAPL", 0.0))
    assert held <= 20_000.0 + 1e-6


def test_concurrent_submits_cannot_overlap_inside_the_broker(tmp_path: Path) -> None:
    inner = FaultBroker()
    broker = _Overlap(inner)
    session = open_fault_session(tmp_path, broker)
    arm(session)
    errors: list[BaseException] = []
    results: list[dict[str, object]] = []

    def _send(index: int, qty: float) -> None:
        try:
            results.append(
                buy(
                    session,
                    quantity=qty,
                    limit_price=10.0,
                    mark=10.0,
                    client_order_id=f"race-{index}",
                    at_ns=TS,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    qtys = [10_000.0, 9_990.0, 9_980.0, 9_970.0, 9_960.0, 9_950.0, 9_940.0, 9_930.0]
    threads = [threading.Thread(target=_send, args=(index, qty)) for index, qty in enumerate(qtys)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert errors == []
    accepted = [row for row in results if row["ok"]]
    assert len(accepted) <= 2
    assert broker.max_inside == 1
    assert broker.submit_count == len(accepted)
    reserved, _cash = session._working_exposure()
    assert reserved.get("AAPL", 0.0) <= 20_000.0 + 1e-6
    assert all(row["broker_called"] is False for row in results if not row["ok"])


def test_unrealized_loss_and_drawdown_latch_and_survive_the_new_york_day(tmp_path: Path) -> None:
    broker = _Fills()
    root = tmp_path / "loss"
    session = open_fault_session(root, broker)
    arm(session)
    bought = buy(session, quantity=2_500.0, client_order_id="long")
    assert bought["ok"] is True
    assert bought["status"] == "filled"
    broker.fill_on_submit = False
    working = buy(
        session,
        symbol="MSFT",
        quantity=10.0,
        client_order_id="still-working",
        at_ns=TS + 1,
    )
    assert working["status"] == "accepted"
    session.mark(symbol="AAPL", price=120.0, actor="ada", at_ns=TS + _STEP)
    assert session.kill_state["latched"] is False
    session.mark(symbol="AAPL", price=96.0, actor="ada", at_ns=TS + 2 * _STEP)
    assert session.kill_state["latched"] is True
    assert session.kill_state["reason"] == "trailing_drawdown"
    parked = session._find("still-working")
    assert parked is not None
    assert parked["status"] == "canceled"
    assert broker.submit_count == 2

    fresh = _Fills()
    root = tmp_path / "daily"
    session = open_fault_session(root, fresh)
    arm(session)
    assert buy(session, quantity=2_500.0, client_order_id="daily")["status"] == "filled"
    session.mark(symbol="AAPL", price=92.0, actor="ada", at_ns=TS + _STEP)
    assert session.kill_state["latched"] is True
    assert session.kill_state["reason"] == "daily_loss"
    cold = open_fault_session(root, fresh)
    assert cold.kill_state["latched"] is True
    thursday = _ns(2024, 1, 4, 0, 30)
    cold.mark(symbol="AAPL", price=92.0, actor="ada", at_ns=thursday)
    assert cold.kill_state["latched"] is True
    monday = _ns(2024, 1, 8, 10, 0)
    cold.mark(symbol="AAPL", price=92.0, actor="ada", at_ns=monday)
    blocked = buy(
        cold,
        quantity=1.0,
        limit_price=92.0,
        mark=92.0,
        client_order_id="next-day",
        at_ns=monday + 1_000_000_000,
    )
    assert blocked["broker_called"] is False
    assert "kill_switch" in blocked["reasons"]
    assert cold.kill_state["latched"] is True
    assert fresh.submit_count == 1


def test_realized_loss_trips_after_the_fill(tmp_path: Path) -> None:
    broker = _Fills()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    assert buy(session, quantity=2_500.0, client_order_id="entry")["status"] == "filled"
    tightened = session.update_limits(_limits(session, max_daily_loss_fraction=0.001), actor="ada")
    assert tightened["applied"] is True
    assert session.kill_state["latched"] is False
    sold = buy(
        session,
        side="sell",
        quantity=2_500.0,
        limit_price=99.5,
        mark=100.0,
        client_order_id="exit",
        at_ns=TS + _STEP,
    )
    assert sold["ok"] is True
    assert sold["status"] == "filled"
    assert session.kill_state["latched"] is True
    assert session.kill_state["reason"] == "daily_loss"
    blocked = buy(
        session,
        quantity=1.0,
        client_order_id="after-loss",
        at_ns=TS + 2 * _STEP,
    )
    assert blocked["broker_called"] is False
    assert "kill_switch" in blocked["reasons"]


def test_short_without_position_or_locate_is_denied_before_the_broker(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    naked = buy(session, side="sell", quantity=1.0, client_order_id="naked")
    assert naked["broker_called"] is False
    assert "reg_sho" in naked["reasons"]
    assert buy(session, quantity=10.0, client_order_id="cover", at_ns=TS + 1)["ok"] is True
    over = buy(
        session,
        side="sell",
        quantity=11.0,
        client_order_id="over",
        at_ns=TS + _STEP,
    )
    assert over["broker_called"] is False
    assert "reg_sho" in over["reasons"]
    assert broker.submit_count == 1

    margin = open_fault_session(tmp_path / "margin", FaultBroker())
    arm(margin)
    broker = margin.broker  # type: ignore[assignment]
    opened = margin.update_limits(
        _limits(margin, cash_account=False), actor="ada", reauthorize=True
    )
    assert opened["applied"] is True
    margin.set_symbol_controls(symbol="AAPL", actor="ada", locate_ok=False, at_ns=TS)
    unlocated = buy(margin, side="sell", quantity=1.0, client_order_id="locate")
    assert unlocated["broker_called"] is False
    assert "reg_sho" in unlocated["reasons"]
    assert broker.submit_count == 0
    margin.set_symbol_controls(symbol="AAPL", actor="ada", locate_ok=True, at_ns=TS + _STEP)
    located = buy(
        margin,
        side="sell",
        quantity=1.0,
        client_order_id="located",
        at_ns=TS + _STEP + 1,
    )
    assert located["broker_called"] is True
    assert broker.submit_count == 1


def test_off_hours_halt_and_precision_stop_before_the_broker(tmp_path: Path) -> None:
    early = _ns(2024, 1, 3, 9, 0)
    broker = FaultBroker()
    session = open_fault_session(tmp_path / "early", broker, now=early)
    arm(session)
    denied = buy(session, client_order_id="early", at_ns=early)
    assert denied["broker_called"] is False
    assert "market_closed" in denied["reasons"]
    assert broker.submit_count == 0

    utc_open = datetime(2024, 1, 3, 14, 0, tzinfo=ZoneInfo("UTC"))
    assert utc_open.astimezone(_ET).hour == 9
    same = int(utc_open.timestamp() * 1_000_000_000)
    utc_broker = FaultBroker()
    utc_session = open_fault_session(tmp_path / "utc", utc_broker, now=same)
    arm(utc_session)
    utc_denied = buy(utc_session, client_order_id="utc", at_ns=same)
    assert utc_denied["broker_called"] is False
    assert "market_closed" in utc_denied["reasons"]

    regular = _ns(2024, 1, 3, 9, 30)
    ok_broker = FaultBroker()
    ok_session = open_fault_session(tmp_path / "open", ok_broker, now=regular)
    arm(ok_session)
    assert buy(ok_session, client_order_id="rth", at_ns=regular)["ok"] is True

    closed = _ns(2024, 1, 3, 16, 0)
    late_broker = FaultBroker()
    late = open_fault_session(tmp_path / "late", late_broker, now=closed)
    arm(late)
    late_denied = buy(late, client_order_id="close", at_ns=closed)
    assert late_denied["broker_called"] is False
    assert "market_closed" in late_denied["reasons"]

    weekend = _ns(2024, 1, 6, 11, 0)
    week_broker = FaultBroker()
    week = open_fault_session(tmp_path / "week", week_broker, now=weekend)
    arm(week)
    week_denied = buy(week, client_order_id="sat", at_ns=weekend)
    assert week_denied["broker_called"] is False
    assert "market_closed" in week_denied["reasons"]

    live = open_fault_session(tmp_path / "halt", FaultBroker())
    arm(live)
    live.set_symbol_controls(symbol="AAPL", actor="ada", halted=True, at_ns=TS)
    halted = buy(live, client_order_id="halted")
    assert halted["broker_called"] is False
    assert "halted" in halted["reasons"]
    cold = open_fault_session(tmp_path / "halt", live.broker)
    halted_again = buy(cold, client_order_id="halted-2", at_ns=TS + 1)
    assert halted_again["broker_called"] is False
    assert "halted" in halted_again["reasons"]

    ticks = open_fault_session(tmp_path / "ticks", FaultBroker())
    arm(ticks)
    bad_qty = buy(ticks, quantity=10.00003, client_order_id="qty-tick")
    assert bad_qty["broker_called"] is False
    assert "qty_precision" in bad_qty["reasons"]
    bad_px = buy(
        ticks,
        limit_price=100.00003,
        client_order_id="px-tick",
        at_ns=TS + 1,
    )
    assert bad_px["broker_called"] is False
    assert "price_precision" in bad_px["reasons"]
    assert ticks.broker.submit_count == 0  # type: ignore[attr-defined]


def test_limit_changes_apply_immediately_and_only_tighten_while_live(tmp_path: Path) -> None:
    broker = FaultBroker()
    root = tmp_path / "caps"
    session = open_fault_session(root, broker)
    arm(session)
    tight = session.update_limits(_limits(session, max_order_quantity=5.0), actor="ada")
    assert tight["applied"] is True
    assert buy(session, quantity=6.0, client_order_id="too-big")["broker_called"] is False
    assert buy(session, quantity=5.0, client_order_id="fits", at_ns=TS + 1)["ok"] is True
    refused = session.update_limits(_limits(session, max_order_quantity=50.0), actor="ada")
    assert refused["applied"] is False
    assert "max_order_quantity" in refused["loosened"]
    assert (
        buy(session, quantity=6.0, client_order_id="still-big", at_ns=TS + _STEP)["broker_called"]
        is False
    )
    structural = session.update_limits(
        _limits(session, ring_capacity=128), actor="ada", reauthorize=True
    )
    assert structural["applied"] is False
    assert "ring_capacity" in structural["structural"]
    opened = session.update_limits(
        _limits(session, max_order_quantity=50.0), actor="ada", reauthorize=True
    )
    assert opened["applied"] is True
    assert opened["reauthorized"] is True
    assert buy(session, quantity=6.0, client_order_id="reauth", at_ns=TS + 2 * _STEP)["ok"] is True

    again = open_fault_session(root, broker)
    assert again.pretrade.limits.max_order_quantity == 50.0
    held = open_fault_session(tmp_path / "held", FaultBroker())
    arm(held)
    held.update_limits(_limits(held, max_order_quantity=4.0), actor="ada")
    resumed = open_fault_session(tmp_path / "held", held.broker)
    assert resumed.pretrade.limits.max_order_quantity == 4.0
    assert buy(resumed, quantity=5.0, client_order_id="remembered")["broker_called"] is False


def test_kill_cancels_every_working_order_and_is_idempotent(tmp_path: Path) -> None:
    broker = _Cancels()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    assert buy(session, quantity=10.0, client_order_id="work-1")["status"] == "accepted"
    assert (
        buy(session, quantity=11.0, client_order_id="work-2", at_ns=TS + 1)["status"] == "accepted"
    )
    first = session.kill(actor="ada", reason="operator halt", at_ns=TS + 2)
    assert first["kill_latched"] is True
    assert first["ok"] is True
    assert set(first["canceled"]) == {"work-1", "work-2"}
    assert first["still_working"] == []
    assert broker.cancel_calls == 2
    assert {row["status"] for row in session.orders} == {"canceled"}
    second = session.kill(actor="ada", reason="operator halt", at_ns=TS + 3)
    assert second["ok"] is True
    assert second["canceled"] == []
    assert broker.cancel_calls == 2
    blocked = buy(session, quantity=1.0, client_order_id="after-kill", at_ns=TS + 4)
    assert blocked["broker_called"] is False
    assert "kill_switch" in blocked["reasons"]


def test_a_rejected_cancel_stays_working_and_kill_can_be_repeated(tmp_path: Path) -> None:
    broker = _RejectsCancel()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    assert buy(session, client_order_id="stuck")["status"] == "accepted"
    first = session.kill(actor="ada", reason="cannot flatten", at_ns=TS + 1)
    assert first["kill_latched"] is True
    assert first["ok"] is False
    assert first["still_working"] == ["stuck"]
    second = session.kill(actor="ada", reason="cannot flatten", at_ns=TS + 2)
    assert second["still_working"] == ["stuck"]
    assert broker.cancel_calls == 2
    assert session.orders[0]["status"] == "accepted"


def test_desk_loss_precision_short_halt_limits_and_cancel_all(tmp_path: Path) -> None:
    desk = open_desk(
        tmp_path / "desk",
        max_daily_loss_fraction=0.02,
        max_trailing_drawdown_fraction=0.05,
        max_name_concentration=1.0,
        max_position_notional=5_000_000.0,
        max_order_notional=1_000_000.0,
    )
    try:
        desk.arm("loopback", "", TS)
        assert (
            desk.submit("AAPL", "BUY", 2_500, 100.0, TS + 1, client_order_id="d-long")["status"]
            == "filled"
        )
        desk.mark("AAPL", 92.0, TS + 2)
        assert desk.engine.book.killed == 1
        assert desk.halt_reason == "daily_loss"
        blocked = desk.submit("AAPL", "BUY", 1, 92.0, TS + 3, client_order_id="d-after")
        assert blocked["accepted"] is False
        assert blocked["kill_latched"] is True
    finally:
        desk.close()

    desk = open_desk(tmp_path / "desk-tick")
    try:
        desk.arm("loopback", "", TS)
        bad = desk.submit("AAPL", "BUY", 10.00003, 100.0, TS + 1, client_order_id="d-tick")
        assert bad["accepted"] is False
        assert bad["transmitted"] is False
        assert "qty_precision" in bad["order"]["reasons"]  # type: ignore[index]
        assert desk._loopback.cancel_sends == 0
        assert desk._loopback._queued == []
        naked = desk.submit("AAPL", "SELL", 1, 100.0, TS + 2, client_order_id="d-short")
        assert naked["accepted"] is False
        assert "reg_sho" in naked["order"]["reasons"]  # type: ignore[index]
        desk.set_symbol_controls("MSFT", TS + 3, halted=True)
        halted = desk.submit("MSFT", "BUY", 1, 100.0, TS + 4, client_order_id="d-halt")
        assert halted["accepted"] is False
        assert "halted" in halted["order"]["reasons"]  # type: ignore[index]
        tight = desk.config.limits.model_copy(update={"max_order_quantity": 4})
        assert desk.update_limits(tight, TS + 5)["applied"] is True
        assert (
            desk.submit("AAPL", "BUY", 5, 100.0, TS + 6, client_order_id="d-five")["accepted"]
            is False
        )
        assert (
            desk.submit("AAPL", "BUY", 4, 100.0, TS + 7, client_order_id="d-four")["status"]
            == "filled"
        )
        loose = desk.config.limits.model_copy(update={"max_order_quantity": 20})
        assert desk.update_limits(loose, TS + 8)["applied"] is False
        assert (
            desk.submit("AAPL", "BUY", 5, 100.0, TS + _STEP, client_order_id="d-still")["accepted"]
            is False
        )
    finally:
        desk.close()

    desk = open_desk(tmp_path / "desk-kill")
    try:
        desk.arm("loopback", "", TS)
        desk._loopback.defer_fill = True
        working = desk.submit("AAPL", "BUY", 8, 100.0, TS + 1, client_order_id="d-work")
        assert working["status"] == "working"
        other = desk.submit("MSFT", "BUY", 7, 100.0, TS + 2, client_order_id="d-work-2")
        assert other["status"] == "working"
        killed = desk.kill("ops", "halt", TS + 3)
        assert killed["kill_latched"] is True
        assert killed["still_working"] == []
        assert set(killed["canceled"]) == {order.order_id for order in desk.orders.values()}
        assert desk._loopback.cancel_sends == 2
        assert {order.lifecycle for order in desk.orders.values()} == {"canceled"}
        desk.kill("ops", "halt", TS + 4)
        assert desk._loopback.cancel_sends == 2
    finally:
        desk.close()
