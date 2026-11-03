"""A retry or a restart must not submit a client order id twice."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.bridge import BridgeError
from quant_fund.live.brokers import BrokerAccount, BrokerAck, FillPage, LiveOrder
from quant_fund.live.config import LiveConfig
from quant_fund.live.desk import LiveDesk, LiveDeskError
from quant_fund.live.errors import LiveGateError
from quant_fund.live.session import open_session
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


class _TimeoutBroker:
    """Accepts nothing. The first submit times out and stays unrecoverable."""

    def __init__(self) -> None:
        self.submits = 0
        self.marks: dict[str, float] = {}

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        return BrokerAccount(
            cash=1_000_000.0, equity=1_000_000.0, positions={}, marks=dict(self.marks)
        )

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.submits += 1
        raise TimeoutError(order.client_order_id)

    def recover(self, client_order_id: str) -> BrokerAck | None:
        del client_order_id
        return None

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        del client_order_id, broker_order_id
        raise TimeoutError("cancel lost")

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        return FillPage(fills=(), truncated=False)


class _CountingBroker:
    def __init__(self) -> None:
        self.submits = 0
        self.marks: dict[str, float] = {}
        self._orders: dict[str, BrokerAck] = {}

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        return BrokerAccount(
            cash=1_000_000.0, equity=1_000_000.0, positions={}, marks=dict(self.marks)
        )

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.submits += 1
        ack = BrokerAck(
            client_order_id=order.client_order_id,
            broker_order_id=f"broker-{order.client_order_id}",
            status="accepted",
            filled_quantity=0.0,
            fill_price=None,
            fills=(),
            reason="",
        )
        self._orders[order.client_order_id] = ack
        return ack

    def recover(self, client_order_id: str) -> BrokerAck | None:
        return self._orders.get(client_order_id)

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        return BrokerAck(
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            status="canceled",
            filled_quantity=0.0,
            fill_price=None,
            fills=(),
            reason="",
        )

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        return FillPage(fills=(), truncated=False)


def _clock() -> int:
    return TS


def _config(tmp: Path) -> LiveConfig:
    return LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "alpaca-paper",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )


def _open(tmp: Path, broker: _TimeoutBroker | _CountingBroker) -> Any:
    return open_session(_config(tmp), PRETRADE, state_dir=tmp, now_ns=_clock, broker=broker)


def _buy(session: Any, **overrides: object) -> Any:
    payload: dict[str, object] = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1.0,
        "limit_price": 100.0,
        "mark": 100.0,
        "actor": "ada",
        "at_ns": TS,
    }
    payload.update(overrides)
    return session.submit(**payload)


def _desk(root: Path, *, fresh: bool) -> LiveDesk:
    cash = 1_000_000.0 if fresh else None
    account = 1_000_000.0 if fresh else None
    return LiveDesk.open(
        root,
        hmac_key=HMAC_KEY,
        config=PretradeConfig(
            schema_version=1,
            limits=limit_config(),
            session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
        ),
        cash=cash,
        account_value=account,
        ts_ns=TS_NS,
    )


def test_timeout_then_a_new_client_id_does_not_submit(tmp_path: Path) -> None:
    broker = _TimeoutBroker()
    session = _open(tmp_path, broker)
    session.arm(actor="ada")
    with pytest.raises(LiveGateError, match="stays inflight"):
        _buy(session, client_order_id="dc-lost")
    assert broker.submits == 1
    with pytest.raises(LiveGateError, match="no order sent"):
        _buy(session, client_order_id="dc-retry")
    replay = _buy(session, client_order_id="dc-lost")
    assert replay["replayed"] is True
    assert broker.submits == 1
    assert [row["client_order_id"] for row in session.orders] == ["dc-lost"]


def test_restart_does_not_submit_the_lost_client_id_again(tmp_path: Path) -> None:
    first = _TimeoutBroker()
    session = _open(tmp_path, first)
    session.arm(actor="ada")
    with pytest.raises(LiveGateError, match="stays inflight"):
        _buy(session)
    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    lost = stored["orders"][0]["client_order_id"]
    assert first.submits == 1
    second = _CountingBroker()
    resumed = _open(tmp_path, second)
    with pytest.raises(LiveGateError, match="no order sent"):
        _buy(resumed, client_order_id="dc-new")
    replay = _buy(resumed, client_order_id=lost)
    assert replay["replayed"] is True
    assert second.submits == 0


def test_desk_records_the_client_id_before_transmit(tmp_path: Path) -> None:
    root = tmp_path / "desk"
    desk = _desk(root, fresh=True)
    sent: list[str] = []
    try:
        desk.arm("loopback", "", TS_NS)

        def crash(_ticket: object) -> None:
            stored = json.loads((root / "state.json").read_text(encoding="utf-8"))
            match = [
                row
                for row in stored["orders"]
                if row["client_order_id"] == "c1" and row["status"] == "unconfirmed"
            ]
            assert match
            assert match[0]["transmitted"] is False
            sent.append("attempt")
            raise RuntimeError("crash")

        desk._loopback.transmit = crash  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="crash"):
            desk.submit("aapl", "BUY", 1, 100.0, TS_NS, client_order_id="c1")
        assert sent == ["attempt"]
    finally:
        desk.close()
    resumed = _desk(root, fresh=False)
    try:

        def blocked(_ticket: object) -> None:
            sent.append("again")

        resumed._loopback.transmit = blocked  # type: ignore[assignment]
        with pytest.raises(LiveDeskError, match="c1 is unconfirmed"):
            resumed.submit("aapl", "BUY", 1, 100.0, TS_NS + 1, client_order_id="c1")
        with pytest.raises(LiveDeskError, match="no order sent"):
            resumed.submit("aapl", "BUY", 1, 100.0, TS_NS + 2, client_order_id="c2")
        assert sent == ["attempt"]
    finally:
        resumed.close()


def test_desk_timeout_does_not_transmit_a_second_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    desk = _desk(tmp_path / "desk", fresh=True)
    sent: list[str] = []
    try:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)

        def timeout(ticket: object) -> None:
            sent.append(str(ticket))
            raise BridgeError("timeout")

        assert desk._drop is not None
        desk._drop.transmit = timeout  # type: ignore[assignment]
        with pytest.raises(LiveDeskError, match="timeout"):
            desk.submit("aapl", "BUY", 1, 100.0, TS_NS, client_order_id="drop-1")
        with pytest.raises(LiveDeskError, match="drop-1 is unconfirmed"):
            desk.submit("aapl", "BUY", 1, 100.0, TS_NS + 1, client_order_id="drop-1")
        with pytest.raises(LiveDeskError, match="no order sent"):
            desk.submit("aapl", "BUY", 1, 100.0, TS_NS + 2, client_order_id="drop-2")
        assert len(sent) == 1
    finally:
        desk.close()
