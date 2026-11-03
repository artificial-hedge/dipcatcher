"""Regressions for restart and degraded-broker gaps. SYNTHETIC, not market evidence.

- a torn audit or desk journal tail must not drop a durable fill or apply it twice
- throttling fails closed after a bounded retry and does not mint a second order
- a future quote must not authorize an earlier order
- an inflight restart reconciles the fills page before a new order is sent
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.live.brokers import BrokerAck, LoopbackBroker
from quant_fund.live.errors import BrokerRejected, LiveGateError
from tests.unit.fault.support import (
    STARTING_CASH,
    TS,
    FaultBroker,
    arm,
    buy,
    open_desk,
    open_fault_session,
)
from tests.unit.pretrade.support import TS_NS


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", lambda _delay: None)


class _PageOnly(FaultBroker):
    def recover(self, client_order_id: str) -> BrokerAck | None:
        ack = super().recover(client_order_id)
        if ack is None:
            return None
        return BrokerAck(
            client_order_id=ack.client_order_id,
            broker_order_id=ack.broker_order_id,
            status="accepted",
            filled_quantity=0.0,
            fill_price=None,
            fills=(),
            reason="",
        )


class _Throttle(FaultBroker):
    def __init__(self, faults: list[BaseException]) -> None:
        super().__init__()
        self.faults = list(faults)
        self.posts: list[str] = []

    def submit(self, order: object) -> BrokerAck:
        client = str(order.client_order_id)
        self.posts.append(client)
        if self.faults:
            raise self.faults.pop(0)
        return super().submit(order)  # type: ignore[arg-type]


def test_torn_journal_reopens_without_losing_or_doubling_the_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = LoopbackBroker(cash=STARTING_CASH)
    root = tmp_path / "audit"
    session = open_fault_session(root, broker, venue="loopback")
    arm(session)
    assert buy(session, quantity=10.0, client_order_id="torn")["status"] == "filled"
    audit = root / "audit.jsonl"
    audit.write_bytes(audit.read_bytes() + b'{"event":"torn"')
    fresh = LoopbackBroker(cash=0.0)
    resumed = open_fault_session(root, fresh, venue="loopback")
    assert fresh.cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert fresh.positions["AAPL"] == pytest.approx(10.0)
    report = resumed.reconcile(at_ns=TS + 1)
    assert report["duplicate_fills"] == 0
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert report["live_pnl_claim"] is False

    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    desk_root = tmp_path / "desk"
    with open_desk(desk_root) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 4, 100.0, TS_NS + 1, client_order_id="desk-torn")
        order = sent["order"]
        assert isinstance(order, dict)
        order_id = str(order["order_id"])
    journal = desk_root / "desk" / "journal.jsonl"
    journal.write_bytes(journal.read_bytes() + b'{"kind":')
    inbound = desk_root / "desk" / "inbound"
    inbound.mkdir()
    (inbound / "fill.json").write_text(
        json.dumps(
            {
                "schema": "live_broker_event.v1",
                "kind": "fill",
                "order_id": order_id,
                "event_id": "fill-1",
                "qty": 4,
                "price": 100,
            }
        ),
        encoding="utf-8",
    )
    with open_desk(desk_root) as desk:
        applied = desk.poll(TS_NS + 2)
    assert applied["positions"] == {"AAPL": 4}
    assert applied["cash"] == pytest.approx(STARTING_CASH - 400.0)
    with open_desk(desk_root) as desk:
        again = desk.poll(TS_NS + 3)
    assert again["cash"] == pytest.approx(STARTING_CASH - 400.0)
    assert again["positions"] == {"AAPL": 4}


def test_throttle_budget_is_idempotent_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", delays.append)
    broker = _Throttle([BrokerRejected("broker HTTP 429: slow down")] * 5)
    session = open_fault_session(tmp_path / "budget", broker, venue="alpaca-paper")
    arm(session)
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=1.0, client_order_id="budget")
    assert broker.posts == ["budget", "budget", "budget"]
    assert delays == [0.01, 0.02]
    assert session.orders[0]["status"] == "inflight"
    assert session.ledger.computed_positions == {}
    assert broker.submit_count == 0

    class _Once(FaultBroker):
        def __init__(self) -> None:
            super().__init__()
            self.posts: list[str] = []
            self._once = True

        def submit(self, order: object) -> BrokerAck:
            client = str(order.client_order_id)
            self.posts.append(client)
            ack = super().submit(order)  # type: ignore[arg-type]
            if self._once:
                self._once = False
                raise TimeoutError("timeout after accept")
            return ack

    once = _Once()
    quiet = open_fault_session(tmp_path / "once", once, venue="alpaca-paper")
    arm(quiet)
    assert buy(quiet, quantity=1.0, client_order_id="once")["status"] == "accepted"
    assert once.posts == ["once"]
    assert once.submit_count == 1


def test_future_quote_cannot_launder_an_earlier_order(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    session.mark(symbol="AAPL", price=150.0, actor="ada", at_ns=TS + 1_000_000_000)
    blocked = buy(
        session,
        quantity=1.0,
        limit_price=150.0,
        mark=100.0,
        client_order_id="skew",
        at_ns=TS + 10,
    )
    assert blocked["reasons"] == ["stale_data"]
    assert blocked["broker_called"] is False
    assert session.marks["AAPL"] == pytest.approx(150.0)
    assert broker.submit_count == 0
    assert session.ledger.computed_cash == pytest.approx(STARTING_CASH)


def test_inflight_restart_reconciles_before_a_new_order(tmp_path: Path) -> None:
    broker = _PageOnly()
    root = tmp_path / "inflight"
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)
    broker.drop_ack = True
    broker.drop_recover = True
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=10.0, client_order_id="open-1")
    assert broker.deliver("open-1", 10.0, 100.0, TS, "fill-open") == "applied"
    broker.drop_ack = False
    broker.drop_recover = False
    resumed = open_fault_session(root, broker, venue="alpaca-paper")
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    resumed.kill(actor="ada", reason="operator halt", at_ns=TS + 1)
    cold = open_fault_session(root, broker, venue="alpaca-paper")
    blocked = buy(cold, quantity=1.0, client_order_id="open-2", at_ns=TS + 2)
    assert blocked["broker_called"] is False
    assert "kill_switch" in blocked["reasons"]
    assert broker.submit_count == 1
    assert cold.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert cold.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert blocked["live_pnl_claim"] is False
