"""Clock skew must not refresh a quote backward. SYNTHETIC timestamps."""

from __future__ import annotations

from pathlib import Path

import pytest
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

from quant_fund.live.desk import LiveDeskError
from quant_fund.live.errors import LiveGateError


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", lambda _delay: None)


def test_future_quote_does_not_authorize_an_earlier_order(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    future = TS + 1_000_000_000
    marked = session.mark(symbol="AAPL", price=150.0, actor="ada", at_ns=future)
    assert marked["ok"] is True
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
    with pytest.raises(LiveGateError, match="older than the stored quote"):
        session.mark(symbol="AAPL", price=90.0, actor="ada", at_ns=TS + 5)
    assert session.marks["AAPL"] == pytest.approx(150.0)
    assert session.ledger.computed_cash == pytest.approx(STARTING_CASH)


def test_desk_future_mark_blocks_an_earlier_submit(tmp_path: Path) -> None:
    with open_desk(tmp_path) as desk:
        future = TS_NS + 1_000_000_000
        desk.mark("AAPL", 150.0, future)
        denied = desk.submit("AAPL", "BUY", 1, 150.0, TS_NS + 10, client_order_id="skew")
        order = denied["order"]
        assert isinstance(order, dict)
        assert denied["accepted"] is False
        assert "stale_data" in order["reasons"]
        assert desk.marks["AAPL"] == pytest.approx(150.0)
        with pytest.raises(LiveDeskError, match="older than the stored quote"):
            desk.mark("AAPL", 90.0, TS_NS + 5)
        assert desk.marks["AAPL"] == pytest.approx(150.0)
        assert desk.positions == {}
        assert denied["live_pnl_claim"] is False


def test_future_fill_cannot_push_the_watermark(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    assert buy(session, quantity=10.0, client_order_id="prints")["status"] == "accepted"
    assert broker.deliver("prints", 4.0, 100.0, TS + 10**18, "far") == "applied"
    first = session.reconcile(at_ns=TS + 5)
    assert first["live_pnl_claim"] is False
    assert session.ledger.fills()[0].fill_time_ns == TS + 5
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(4.0)
    cash = session.ledger.computed_cash
    assert broker.deliver("prints", 4.0, 100.0, TS, "back") == "applied"
    second = session.reconcile(at_ns=TS + 6)
    assert second["out_of_order_fills"] == 1
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(8.0)
    assert session.ledger.computed_cash == pytest.approx(cash - 400.0)
    third = session.reconcile(at_ns=TS + 7)
    assert third["duplicate_fills"] == 3
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(8.0)
    assert session.ledger.computed_cash == pytest.approx(cash - 400.0)
    assert broker.cash == pytest.approx(session.ledger.computed_cash)


def test_fill_inside_the_age_window_keeps_its_timestamp(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    assert buy(session, quantity=2.0, client_order_id="near")["status"] == "accepted"
    stamp = TS + 1_000
    assert broker.deliver("near", 2.0, 100.0, stamp, "near-fill") == "applied"
    session.reconcile(at_ns=TS)
    assert session.ledger.fills()[0].fill_time_ns == stamp
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(2.0)
