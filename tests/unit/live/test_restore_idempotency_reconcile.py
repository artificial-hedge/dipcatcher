"""Corrupt restore, idempotent submit, and position drift. SYNTHETIC."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.config.models import KillSwitchConfig
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.live.brokers import BrokerAck, LiveOrder
from quant_fund.live.desk import LiveDesk, LiveDeskError
from quant_fund.live.errors import LiveGateError
from quant_fund.monitoring.kill_switch import ENABLED, HALT_NEW_ORDERS, KillSwitch
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from quant_fund.schemas.errors import KillSwitchActive
from tests.property._books import research_config
from tests.unit.fault.support import FaultBroker, arm, buy, open_desk, open_fault_session
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config


class LateAckBroker(FaultBroker):
    """Stores the order, withholds the ack, then returns that same order."""

    def __init__(self) -> None:
        super().__init__()
        self.posts: list[str] = []
        self.release = False

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.posts.append(order.client_order_id)
        if order.client_order_id not in self._acks:
            super().submit(order)
        if not self.release:
            raise TimeoutError("ack late")
        return self._acks[order.client_order_id]

    def recover(self, client_order_id: str) -> BrokerAck | None:
        if not self.release:
            return None
        return super().recover(client_order_id)


class DoubleAckBroker(FaultBroker):
    """A second ack for the same key returns the original broker order."""

    def __init__(self) -> None:
        super().__init__()
        self.posts: list[str] = []
        self.acks = 0

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.posts.append(order.client_order_id)
        self.acks += 1
        return super().submit(order)


def _seed(tmp: Path) -> tuple[object, FaultBroker]:
    broker = FaultBroker()
    session = open_fault_session(tmp, broker)
    arm(session)
    filled = buy(session, quantity=1.0, client_order_id="seed")
    assert filled["ok"] is True
    return session, broker


def test_unknown_set_state_halts_an_enabled_switch() -> None:
    switch = KillSwitch(KillSwitchConfig(state=ENABLED))
    with pytest.raises(KillSwitchActive, match="invalid"):
        switch.set_state("NOT_A_STATE")
    assert switch.state == HALT_NEW_ORDERS
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()


def test_from_state_rejects_an_unknown_kill_and_does_not_return_enabled(tmp_path: Path) -> None:
    cfg = research_config()
    broker = SimulatedBroker(config=cfg, initial_cash=1_000.0)
    broker.kill.set_state(ENABLED)
    state = broker.to_dict()
    state["kill_state"] = "NOT_A_KILL_STATE"
    with pytest.raises(ValueError, match="restored kill switch state is invalid"):
        SimulatedBroker.from_state(cfg, state)
    with pytest.raises(KillSwitchActive):
        broker.kill.set_state("ALSO_BAD")
    assert broker.kill.state == HALT_NEW_ORDERS


def test_truncated_session_state_halts_and_raises(tmp_path: Path) -> None:
    session, broker = _seed(tmp_path)
    (tmp_path / "state.json").write_text("{", encoding="utf-8")
    with pytest.raises(LiveGateError, match="truncated"):
        session.status()
    assert session.kill_state["latched"] is True
    assert session.kill_state["state"] == HALT_NEW_ORDERS
    with pytest.raises(LiveGateError):
        buy(session, client_order_id="after-truncate")
    assert broker.submit_count == 1


def test_corrupt_session_state_halts_and_raises(tmp_path: Path) -> None:
    session, _broker = _seed(tmp_path)
    (tmp_path / "state.json").write_text("[]", encoding="utf-8")
    with pytest.raises(LiveGateError, match="corrupt"):
        session.status()
    assert session.kill_state["latched"] is True
    assert session.kill_state["state"] == HALT_NEW_ORDERS


def test_unknown_kill_name_in_the_state_file_halts(tmp_path: Path) -> None:
    session, _broker = _seed(tmp_path)
    path = tmp_path / "state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["kill"]["state"] = "NOT_A_STATE"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LiveGateError, match="unknown kill switch state"):
        session.status()
    assert session.kill_state["latched"] is True
    assert session.kill_state["state"] == HALT_NEW_ORDERS
    with pytest.raises(LiveGateError):
        buy(session, client_order_id="after-unknown")


def test_truncated_desk_state_raises(tmp_path: Path) -> None:
    desk = open_desk(tmp_path)
    desk._save()
    desk.close()
    path = tmp_path / "desk" / "state.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(LiveDeskError, match="truncated"):
        LiveDesk.open(
            tmp_path / "desk",
            hmac_key=HMAC_KEY,
            config=PretradeConfig(
                schema_version=1,
                limits=limit_config(),
                session=SessionConfig(
                    timezone="America/New_York", open_minute=570, close_minute=960
                ),
            ),
            ts_ns=TS_NS,
        )


def test_late_ack_and_restart_do_not_create_a_second_order(tmp_path: Path) -> None:
    broker = LateAckBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    with pytest.raises(LiveGateError, match="inflight"):
        buy(session, quantity=1.0, client_order_id="late-1")
    assert broker.posts == ["late-1", "late-1", "late-1"]
    assert list(broker.orders) == ["late-1"]
    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert stored["orders"][0]["idempotency_key"] == "late-1"
    broker.release = True
    resumed = open_fault_session(tmp_path, broker)
    arm(resumed)
    found = buy(resumed, quantity=1.0, client_order_id="late-1")
    assert found["status"] == "accepted"
    assert found["replayed"] is True
    assert list(broker.orders) == ["late-1"]
    assert broker.posts == ["late-1", "late-1", "late-1"]


def test_a_second_ack_does_not_create_a_second_order(tmp_path: Path) -> None:
    broker = DoubleAckBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    first = buy(session, quantity=2.0, client_order_id="twice")
    assert first["ok"] is True
    assert first["idempotency_key"] == "twice"
    again = buy(session, quantity=2.0, client_order_id="twice")
    assert again["replayed"] is True
    order = LiveOrder("twice", "AAPL", "buy", 2.0, 100.0, 1)
    second_ack = broker.submit(order)
    assert second_ack.broker_order_id == first["broker_order_id"]
    assert broker.acks == 2
    assert list(broker.orders) == ["twice"]
    assert broker.posts == ["twice", "twice"]


def test_position_mismatch_halts_new_orders_and_alerts(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    filled = buy(session, quantity=4.0, client_order_id="pos-1")
    assert filled["ok"] is True
    broker.positions["AAPL"] = 1.0
    report = session.reconcile(at_ns=None)
    assert report["ok"] is False
    assert "position mismatch AAPL" in report["alerts"]
    assert session.kill_state["latched"] is True
    assert session.kill_state["state"] == HALT_NEW_ORDERS
    blocked = buy(session, quantity=1.0, client_order_id="pos-2")
    assert blocked["ok"] is False
    assert blocked["broker_called"] is False
    assert "kill_switch" in blocked["reasons"]


def test_a_move_away_from_the_last_mark_does_not_replace_it(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    assert buy(session, quantity=1.0, limit_price=100.0, mark=100.0, client_order_id="mark")["ok"]
    denied = buy(
        session,
        quantity=1.0,
        limit_price=150.0,
        mark=150.0,
        client_order_id="wide",
    )
    assert denied["ok"] is False
    assert denied["broker_called"] is False
    assert "price_collar" in denied["reasons"]
    assert session.marks["AAPL"] == 100.0
    assert broker.submit_count == 1
