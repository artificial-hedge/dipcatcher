"""Crash cuts between submit, fill, and the durable ledger. No live broker."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.unit.fault.support import (
    STARTING_CASH,
    TS,
    FaultBroker,
    arm,
    buy,
    open_fault_session,
)

from quant_fund.execution.order_recon import Fill, Order
from quant_fund.live.brokers import LoopbackBroker
from quant_fund.live.errors import LiveGateError
from quant_fund.paper.broker_adapter import PaperBrokerAdapter

T0 = datetime(2024, 1, 3, 20, 0, tzinfo=UTC)


def test_restart_applies_the_broker_fill_the_ledger_missed(tmp_path: Path) -> None:
    """Process dies after the broker fill and before the session's second write."""
    broker = FaultBroker()
    broker.drop_ack = True
    broker.drop_recover = True
    root = tmp_path / "cut"
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=10.0, client_order_id="cut-1")
    assert broker.deliver("cut-1", 10.0, 100.0, TS, "fill-cut") == "applied"
    assert session.ledger.computed_positions == {}
    assert session.ledger.computed_cash == pytest.approx(STARTING_CASH)
    broker.drop_ack = False
    broker.drop_recover = False
    resumed = open_fault_session(root, broker, venue="alpaca-paper")
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert broker.cash == pytest.approx(resumed.ledger.computed_cash)
    assert broker.positions["AAPL"] == pytest.approx(resumed.ledger.computed_positions["AAPL"])
    report = resumed.reconcile(at_ns=TS + 1)
    assert report["ok"] is True
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert broker.submit_count == 1
    assert report["live_pnl_claim"] is False


def test_loopback_restart_rebuilds_the_broker_from_durable_state(tmp_path: Path) -> None:
    broker = LoopbackBroker(cash=STARTING_CASH)
    root = tmp_path / "loop"
    session = open_fault_session(root, broker, venue="loopback")
    arm(session)
    filled = buy(session, quantity=10.0, client_order_id="loop-1")
    assert filled["status"] == "filled"
    cash = session.ledger.computed_cash
    positions = dict(session.ledger.computed_positions)
    fresh = LoopbackBroker(cash=0.0)
    resumed = open_fault_session(root, fresh, venue="loopback")
    assert fresh.cash == pytest.approx(cash)
    assert fresh.positions["AAPL"] == pytest.approx(positions["AAPL"])
    assert resumed.ledger.computed_cash == pytest.approx(cash)
    report = resumed.reconcile(at_ns=TS + 1)
    assert report["ok"] is True
    assert resumed.ledger.computed_cash == pytest.approx(cash)
    assert fresh.positions["AAPL"] == pytest.approx(10.0)
    assert report["live_pnl_claim"] is False


def test_paper_crash_before_checkpoint_submits_once(tmp_path: Path) -> None:
    path = tmp_path / "paper.json"
    run = PaperBrokerAdapter(state_path=path, endpoint_kind="paper", starting_cash=1_000.0)
    assert run.endpoint_kind == "paper"
    first = Order(order_id="o1", security_id="SEC_A", side="BUY", quantity=10.0)
    second = Order(order_id="o2", security_id="SEC_A", side="BUY", quantity=4.0)
    assert run.submit(first) == "accepted"  # type: ignore[arg-type]
    run.checkpoint()
    assert run.submit(second) == "accepted"  # type: ignore[arg-type]
    resumed, report = PaperBrokerAdapter.resume(path, endpoint_kind="paper")
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    assert resumed.submit(first) == "duplicate_order"  # type: ignore[arg-type]
    assert resumed.submit(second) == "accepted"  # type: ignore[arg-type]
    assert resumed.submit(second) == "duplicate_order"  # type: ignore[arg-type]
    assert resumed.cash == pytest.approx(1_000.0)
    assert resumed.position("SEC_A") == pytest.approx(0.0)


def test_paper_crash_after_fill_before_checkpoint_reapplies_once(tmp_path: Path) -> None:
    path = tmp_path / "paper.json"
    run = PaperBrokerAdapter(state_path=path, endpoint_kind="paper", starting_cash=1_000.0)
    order = Order(order_id="o1", security_id="SEC_A", side="BUY", quantity=10.0)
    fill = Fill(
        fill_id="f1",
        order_id="o1",
        security_id="SEC_A",
        quantity=10.0,
        price=10.0,
        fill_time=T0,
    )
    assert run.submit(order) == "accepted"  # type: ignore[arg-type]
    run.checkpoint()
    assert run.apply_fill(fill) == "accepted"
    assert run.cash == pytest.approx(900.0)
    assert run.position("SEC_A") == pytest.approx(10.0)
    resumed, report = PaperBrokerAdapter.resume(path, endpoint_kind="paper")
    assert report["ok"] is True
    assert report["live_pnl_claim"] is False
    assert resumed.cash == pytest.approx(1_000.0)
    assert resumed.position("SEC_A") == pytest.approx(0.0)
    assert resumed.apply_fill(fill) == "accepted"
    assert resumed.cash == pytest.approx(900.0)
    assert resumed.position("SEC_A") == pytest.approx(10.0)
    assert resumed.apply_fill(fill) == "duplicate_fill"
    assert resumed.cash == pytest.approx(900.0)
    assert resumed.position("SEC_A") == pytest.approx(10.0)
