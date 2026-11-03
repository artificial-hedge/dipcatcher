"""Restart adopts broker truth before the next order. SYNTHETIC, no sockets."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.fault.support import STARTING_CASH, TS, FaultBroker, arm, buy, open_fault_session

from quant_fund.live.brokers import BrokerAck
from quant_fund.live.errors import LiveGateError


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", lambda _delay: None)


class _PageOnly(FaultBroker):
    """The ack is accepted and empty. The print lives only on the fills page."""

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


def _inflight(root: Path, broker: _PageOnly, client: str) -> None:
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)
    broker.drop_ack = True
    broker.drop_recover = True
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=10.0, client_order_id=client)
    assert broker.submit_count == 1
    assert broker.deliver(client, 10.0, 100.0, TS, f"fill-{client}") == "applied"
    broker.drop_ack = False
    broker.drop_recover = False


def test_inflight_restart_adopts_the_fill_before_a_new_order(tmp_path: Path) -> None:
    broker = _PageOnly()
    root = tmp_path / "inflight"
    _inflight(root, broker, "open-1")
    resumed = open_fault_session(root, broker, venue="alpaca-paper")
    assert resumed.orders[0]["status"] == "filled"
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert broker.cash == pytest.approx(resumed.ledger.computed_cash)
    second = buy(resumed, quantity=1.0, client_order_id="open-2", at_ns=TS + 1)
    assert second["status"] == "accepted"
    assert second["broker_called"] is True
    assert broker.submit_count == 2
    assert list(broker.orders) == ["open-1", "open-2"]
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert resumed.orders[0]["status"] == "filled"
    again = open_fault_session(root, broker, venue="alpaca-paper")
    assert again.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert again.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert broker.submit_count == 2
    assert second["live_pnl_claim"] is False


def test_kill_latch_survives_restart_and_blocks_before_the_broker(tmp_path: Path) -> None:
    broker = _PageOnly()
    root = tmp_path / "halt"
    _inflight(root, broker, "halt-1")
    session = open_fault_session(root, broker, venue="alpaca-paper")
    session.kill(actor="ada", reason="operator halt", at_ns=TS + 1)
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    cold = open_fault_session(root, broker, venue="alpaca-paper")
    assert cold.kill_state["latched"] is True
    assert cold.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert cold.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    blocked = buy(cold, quantity=1.0, client_order_id="halt-2", at_ns=TS + 2)
    assert blocked["ok"] is False
    assert blocked["broker_called"] is False
    assert "kill_switch" in blocked["reasons"]
    assert broker.submit_count == 1
    assert cold.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert cold.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert blocked["live_pnl_claim"] is False


def test_same_client_retry_does_not_reconcile_into_a_second_submit(tmp_path: Path) -> None:
    broker = _PageOnly()
    root = tmp_path / "same"
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)
    broker.drop_ack = True
    broker.drop_recover = True
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=1.0, client_order_id="same-1")
    replay = buy(session, quantity=1.0, client_order_id="same-1", at_ns=TS + 1)
    assert replay["replayed"] is True
    assert replay["status"] == "inflight"
    assert broker.submit_count == 1
