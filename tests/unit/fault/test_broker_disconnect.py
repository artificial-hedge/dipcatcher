"""Broker disconnect mid-order. SYNTHETIC loopback stand-in; no alpaca-live socket."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.unit.fault.support import (
    SEEDS,
    STARTING_CASH,
    TS,
    FaultBroker,
    Scheduler,
    arm,
    buy,
    open_fault_session,
)

from quant_fund.live.brokers import LiveOrder
from quant_fund.live.errors import LiveGateError


def test_scheduler_replays_one_interleaving() -> None:
    left = Scheduler(11).interleave(["deliver", "retry", "unknown"])
    right = Scheduler(11).interleave(["deliver", "retry", "unknown"])
    assert left == right


def test_lost_ack_and_late_fill_do_not_double_submit(tmp_path: Path) -> None:
    """Each seed delivers the fill and retries while the ack is still dropped."""
    for seed in SEEDS[:4]:
        _disconnect(tmp_path / f"s{seed}", seed)


def _disconnect(root: Path, seed: int) -> None:
    broker = FaultBroker()
    broker.connected = False
    broker.drop_ack = True
    broker.drop_recover = True
    names = Scheduler(seed).interleave(["deliver", "retry"])
    session = open_fault_session(root, broker, venue="alpaca-paper")
    assert session.venue == "alpaca-paper"
    arm(session)
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=10.0, client_order_id=f"lost-{seed}")
    assert broker.submit_count == 1
    stored = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert stored["orders"][0]["status"] == "inflight"
    for name in names:
        if name == "retry":
            replay = buy(session, quantity=10.0, client_order_id=f"lost-{seed}")
            assert replay["replayed"] is True
            assert replay["status"] == "inflight"
            assert replay["broker_called"] is True
        else:
            assert broker.deliver(f"lost-{seed}", 10.0, 100.0, TS, f"fill-{seed}") == "applied"
    assert broker.submit_count == 1
    assert session.ledger.computed_positions == {}
    broker.connected = True
    broker.drop_ack = False
    broker.drop_recover = False
    resumed = open_fault_session(root, broker, venue="alpaca-paper")
    assert resumed.orders[0]["status"] == "filled"
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert broker.cash == pytest.approx(resumed.ledger.computed_cash)
    assert broker.positions["AAPL"] == pytest.approx(10.0)
    again = buy(resumed, quantity=10.0, client_order_id=f"lost-{seed}", at_ns=TS + 1)
    assert again["replayed"] is True
    assert broker.submit_count == 1
    broker.submit(
        LiveOrder(
            client_order_id=f"lost-{seed}",
            symbol="AAPL",
            side="buy",
            quantity=10.0,
            limit_price=100.0,
            fill_time_ns=TS,
        )
    )
    assert broker.submit_count == 1
    assert again["live_pnl_claim"] is False


def test_unknown_broker_order_trips_the_halt(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    broker.drop_ack = True
    broker.drop_recover = True
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=2.0, client_order_id="known-1")
    assert broker.deliver("known-1", 2.0, 100.0, TS, "fill-known") == "applied"
    broker.drop_ack = False
    broker.drop_recover = False
    resumed = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(2.0)
    cash = resumed.ledger.computed_cash
    broker.inject_unknown(fill_id="ghost-1", quantity=5.0, price=100.0, ts=TS + 1)
    report = resumed.reconcile(at_ns=TS + 2)
    assert report["ok"] is False
    assert report["orphan_fills"] == 1
    assert report["live_pnl_claim"] is False
    assert resumed.kill_state["latched"] is True
    assert "orphan_fill" in str(resumed.kill_state["reason"])
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(2.0)
    assert resumed.ledger.computed_cash == pytest.approx(cash)
    assert broker.positions["AAPL"] == pytest.approx(2.0)
    blocked = buy(resumed, quantity=1.0, client_order_id="after-halt", at_ns=TS + 3)
    assert blocked["ok"] is False
    assert blocked["broker_called"] is False
    assert "kill_switch" in blocked["reasons"]
    assert broker.submit_count == 1
    cold = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    assert cold.kill_state["latched"] is True
    assert cold.ledger.computed_positions["AAPL"] == pytest.approx(2.0)
    still = buy(cold, quantity=1.0, client_order_id="after-restart", at_ns=TS + 4)
    assert still["ok"] is False
    assert "kill_switch" in still["reasons"]
    assert broker.submit_count == 1
