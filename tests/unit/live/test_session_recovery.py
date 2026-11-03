from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import (
    BrokerAccount,
    BrokerAck,
    BrokerFill,
    FillPage,
    LiveOrder,
    LoopbackBroker,
)
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import LiveGateError
from quant_fund.live.risk_memory import export_pretrade
from quant_fund.live.session import LiveSession, open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def _open(tmp: Path, *, venue: str = "loopback", broker: Any = None) -> LiveSession:
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": venue,
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    return open_session(
        config,
        PRETRADE,
        state_dir=tmp,
        now_ns=lambda: TS,
        broker=broker,
    )


def _submit(session: LiveSession, **overrides: object) -> dict[str, Any]:
    payload: dict[str, object] = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1.0,
        "limit_price": 100.0,
        "mark": 100.0,
        "actor": "ada",
        "client_order_id": "dc-retry",
        "at_ns": TS,
    }
    payload.update(overrides)
    return session.submit(**payload)


def _snapshot(session: LiveSession) -> dict[str, Any]:
    return {
        "orders": deepcopy(session.orders),
        "marks": dict(session.marks),
        "ledger_cash": session.ledger.computed_cash,
        "ledger_positions": deepcopy(session.ledger.computed_positions),
        "fills": session.ledger.fills(),
        "account": session.broker.account(),
        "state": session.state_path.read_bytes(),
    }


class _CountingLoopbackBroker(LoopbackBroker):
    def __init__(self) -> None:
        super().__init__(cash=1_000_000.0)
        self.defer_fill = True
        self.submit_calls = 0

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.submit_calls += 1
        return super().submit(order)


class _RecoveryBroker:
    def __init__(self, recovered: BrokerAck | None = None) -> None:
        self.recovered = recovered
        self.submit_calls = 0
        self.recover_calls = 0
        self.marks: dict[str, float] = {}

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def submit(self, order: LiveOrder) -> BrokerAck:
        del order
        self.submit_calls += 1
        raise TimeoutError("response lost")

    def recover(self, client_order_id: str) -> BrokerAck | None:
        del client_order_id
        self.recover_calls += 1
        return self.recovered

    def account(self) -> BrokerAccount:
        return BrokerAccount(
            cash=1_000_000.0,
            equity=1_000_000.0,
            positions={},
            marks=dict(self.marks),
        )

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        del client_order_id, broker_order_id
        raise AssertionError("cancel is not expected")

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        return FillPage(fills=(), truncated=False)


def _filled_ack(*, client_order_id: str, broker_order_id: str) -> BrokerAck:
    fill = BrokerFill(
        fill_id=f"fill:{client_order_id}",
        client_order_id=client_order_id,
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        price=100.0,
        fill_time_ns=TS,
    )
    return BrokerAck(
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        status="filled",
        filled_quantity=1.0,
        fill_price=100.0,
        fills=(fill,),
        reason="",
    )


def test_identical_retry_ignores_actor_mark_and_timestamp(tmp_path: Path) -> None:
    broker = _CountingLoopbackBroker()
    session = _open(tmp_path, broker=broker)
    session.arm(actor="ada")
    first = _submit(session)
    before = _snapshot(session)
    before_risk = export_pretrade(session.engine)

    replay = _submit(session, actor="grace", mark=150.0, at_ns=TS + 1)

    assert first["replayed"] is False
    assert replay["replayed"] is True
    assert replay["status"] == first["status"]
    assert broker.submit_calls == 1
    assert _snapshot(session) == before
    assert export_pretrade(session.engine) == before_risk


@pytest.mark.parametrize(
    "changed",
    [
        {"symbol": "MSFT"},
        {"side": "sell"},
        {"quantity": 2.0},
        {"limit_price": 101.0},
    ],
    ids=["symbol", "side", "quantity", "limit_price"],
)
def test_conflicting_retry_leaves_session_and_broker_untouched(
    tmp_path: Path, changed: dict[str, object]
) -> None:
    broker = _CountingLoopbackBroker()
    session = _open(tmp_path, broker=broker)
    session.arm(actor="ada")
    _submit(session)
    before = _snapshot(session)
    before_risk = export_pretrade(session.engine)

    with pytest.raises(LiveGateError, match="already associated with a different order"):
        _submit(session, **changed)

    assert broker.submit_calls == 1
    assert _snapshot(session) == before
    assert export_pretrade(session.engine) == before_risk


def test_conflicting_inflight_retry_fails_before_recovery(tmp_path: Path) -> None:
    broker = _RecoveryBroker()
    first = _open(tmp_path, venue="alpaca-paper", broker=broker)
    first.arm(actor="ada")
    with pytest.raises(LiveGateError, match="stays inflight"):
        _submit(first)

    resumed = _open(tmp_path, venue="alpaca-paper", broker=broker)
    before = _snapshot(resumed)
    before_risk = export_pretrade(resumed.engine)
    recover_calls = broker.recover_calls

    with pytest.raises(LiveGateError, match="already associated with a different order"):
        _submit(resumed, quantity=2.0)

    assert broker.submit_calls == 1
    assert broker.recover_calls == recover_calls
    assert _snapshot(resumed) == before
    assert export_pretrade(resumed.engine) == before_risk


def test_send_rejects_foreign_recovery_without_resubmitting(tmp_path: Path) -> None:
    broker = _RecoveryBroker(
        _filled_ack(client_order_id="dc-foreign", broker_order_id="broker-foreign")
    )
    session = _open(tmp_path, venue="alpaca-paper", broker=broker)
    session.arm(actor="ada")

    with pytest.raises(LiveGateError, match="broker ack client_order_id does not match"):
        _submit(session)

    before = _snapshot(session)
    assert session.orders[0]["status"] == "inflight"
    assert session.orders[0]["broker_order_id"] == ""
    assert session.ledger.fills() == []
    stored = json.loads(session.state_path.read_text(encoding="utf-8"))
    assert stored["orders"][0]["status"] == "inflight"

    with pytest.raises(LiveGateError, match="broker ack client_order_id does not match"):
        _submit(session)

    assert broker.submit_calls == 1
    assert _snapshot(session) == before


@pytest.mark.parametrize("mismatch", ["client_order_id", "broker_order_id"])
def test_resumed_mismatched_recovery_leaves_order_inflight_without_resubmitting(
    tmp_path: Path, mismatch: str
) -> None:
    broker = _RecoveryBroker()
    first = _open(tmp_path, venue="alpaca-paper", broker=broker)
    first.arm(actor="ada")
    with pytest.raises(LiveGateError, match="stays inflight"):
        _submit(first)

    if mismatch == "broker_order_id":
        first.orders[0]["broker_order_id"] = "broker-known"
        first._persist()
    resumed = _open(tmp_path, venue="alpaca-paper", broker=broker)
    broker.recovered = _filled_ack(
        client_order_id="dc-foreign" if mismatch == "client_order_id" else "dc-retry",
        broker_order_id=("broker-foreign" if mismatch == "broker_order_id" else "broker-recovered"),
    )
    before = _snapshot(resumed)

    with pytest.raises(LiveGateError, match=f"broker ack {mismatch} does not match"):
        _submit(resumed)

    assert broker.submit_calls == 1
    assert resumed.orders[0]["status"] == "inflight"
    assert resumed.ledger.fills() == []
    assert _snapshot(resumed) == before
    stored = json.loads(resumed.state_path.read_text(encoding="utf-8"))
    assert stored["orders"][0]["status"] == "inflight"
