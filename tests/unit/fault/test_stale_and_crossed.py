"""Stale and crossed quotes while an order is working. No live broker."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.fault.support import STARTING_CASH, TS, arm, buy, open_desk, open_fault_session
from tests.unit.pretrade.support import TS_NS

from quant_fund.live.brokers import LoopbackBroker


def test_stale_quote_blocks_new_orders_and_cancel_until_mark(tmp_path: Path) -> None:
    broker = LoopbackBroker(cash=STARTING_CASH)
    broker.defer_fill = True
    session = open_fault_session(tmp_path, broker, venue="loopback")
    arm(session)
    working = buy(session, quantity=10.0, client_order_id="work-1")
    assert working["status"] == "accepted"
    assert working["broker_called"] is True
    age = int(session.pretrade.limits.max_mark_age_ns)
    late = TS + age + 1
    blocked = buy(
        session,
        quantity=1.0,
        client_order_id="stale-1",
        at_ns=late,
    )
    assert blocked["ok"] is False
    assert blocked["reasons"] == ["stale_data"]
    assert blocked["broker_called"] is False
    assert "stale-1" not in broker._orders
    assert session.orders[0]["status"] == "accepted"
    denied = session.cancel(client_order_id="work-1", actor="ada", at_ns=late)
    assert denied["ok"] is False
    assert denied["reasons"] == ["stale_data"]
    assert denied["broker_called"] is False
    assert session.orders[0]["status"] == "accepted"
    session.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=late)
    canceled = session.cancel(client_order_id="work-1", actor="ada", at_ns=late)
    assert canceled["status"] == "canceled"
    assert canceled["broker_called"] is True
    fresh = buy(session, quantity=2.0, client_order_id="fresh-1", at_ns=late)
    assert fresh["status"] == "accepted"
    crossed = buy(
        session,
        quantity=1.0,
        limit_price=110.0,
        mark=100.0,
        client_order_id="cross-1",
        at_ns=late + 1,
    )
    assert crossed["ok"] is False
    assert "price_collar" in crossed["reasons"]
    assert crossed["broker_called"] is False
    assert session._find("fresh-1")["status"] == "accepted"
    assert "cross-1" not in broker._orders
    assert crossed["live_pnl_claim"] is False


def test_desk_stale_mark_blocks_a_new_order_and_keeps_the_working_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    with open_desk(tmp_path) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 4, 50.0, TS_NS + 1, client_order_id="open-1")
        order = sent["order"]
        assert isinstance(order, dict)
        assert sent["status"] == "working"
        age = int(desk.config.limits.max_mark_age_ns)
        late = TS_NS + 1 + age + 1
        blocked = desk.submit("aapl", "BUY", 1, 50.0, late, client_order_id="stale-1")
        assert blocked["accepted"] is False
        denied_order = blocked["order"]
        assert isinstance(denied_order, dict)
        assert denied_order["reasons"] == ["stale_data"]
        assert desk.orders[str(order["order_id"])].lifecycle == "new"
        outbound = list((tmp_path / "desk" / "outbound").glob("*.json"))
        assert len(outbound) == 1
        held = desk.cancel(str(order["order_id"]), late)
        assert held["accepted"] is False
        assert "stale_data" in list(held["reasons"])  # type: ignore[arg-type]
        assert not (tmp_path / "desk" / "outbound" / f"cancel-{order['order_id']}.json").exists()
        desk.mark("aapl", 50.0, late)
        released = desk.cancel(str(order["order_id"]), late)
        assert released["accepted"] is True
        assert (tmp_path / "desk" / "outbound" / f"cancel-{order['order_id']}.json").exists()
        assert desk.orders[str(order["order_id"])].lifecycle == "new"
        assert desk.status()["positions"] == {}
        assert blocked["live_pnl_claim"] is False
