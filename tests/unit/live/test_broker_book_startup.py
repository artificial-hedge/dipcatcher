"""Startup halts when the broker book is missing or disagrees with the ledger."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import BrokerAccount, BrokerAck, FillPage, LiveOrder
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import LiveGateError
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


class _BookBroker:
    def __init__(
        self,
        *,
        cash: float = 1_000_000.0,
        positions: dict[str, float] | None = None,
        fail: bool = False,
    ) -> None:
        self.cash = cash
        self.positions = {} if positions is None else dict(positions)
        self.fail = fail
        self.submits = 0
        self.marks: dict[str, float] = {}

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        if self.fail:
            raise TimeoutError("account snapshot unavailable")
        return BrokerAccount(
            cash=self.cash,
            equity=max(self.cash, 1.0),
            positions=dict(self.positions),
            marks=dict(self.marks),
        )

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.submits += 1
        return BrokerAck(
            client_order_id=order.client_order_id,
            broker_order_id=f"broker-{order.client_order_id}",
            status="accepted",
            filled_quantity=0.0,
            fill_price=None,
            fills=(),
            reason="",
        )

    def recover(self, client_order_id: str) -> BrokerAck | None:
        del client_order_id
        return None

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


def _config(tmp: Path, **overrides: object) -> LiveConfig:
    payload: dict[str, object] = {
        "schema_version": 1,
        "venue": "alpaca-paper",
        "starting_cash": 1_000_000.0,
        "calendar": "us_rth",
        "pretrade_config": str(PRETRADE),
        "state_dir": str(tmp),
        "cash_tolerance": 1e-6,
        "position_tolerance": 1e-6,
    }
    payload.update(overrides)
    return LiveConfig.model_validate(payload)


def _open(tmp: Path, broker: _BookBroker, **overrides: object) -> Any:
    return open_session(
        _config(tmp, **overrides),
        PRETRADE,
        state_dir=tmp,
        now_ns=_clock,
        broker=broker,
    )


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


def test_missing_snapshot_halts_and_sends_nothing(tmp_path: Path) -> None:
    broker = _BookBroker(fail=True)
    session = _open(tmp_path, broker)
    status = session.status()
    assert status["kill_latched"] is True
    assert status["halt_reason"] == ("broker account snapshot could not be fetched; no new orders")
    session.arm(actor="ada")
    with pytest.raises(LiveGateError, match="no new orders"):
        _buy(session, client_order_id="dc-missing")
    assert broker.submits == 0
    assert session.orders == []


def test_position_disagreement_on_restart_sends_nothing(tmp_path: Path) -> None:
    first = _BookBroker()
    session = _open(tmp_path, first)
    session.arm(actor="ada")
    assert _buy(session, client_order_id="dc-book")["status"] == "accepted"
    assert first.submits == 1
    first.positions["AAPL"] = 4.0
    resumed = _open(tmp_path, first)
    status = resumed.status()
    assert status["kill_latched"] is True
    assert "disagree with the local ledger" in status["halt_reason"]
    assert "no new orders" in status["halt_reason"]
    with pytest.raises(LiveGateError, match="no new orders"):
        _buy(resumed, client_order_id="dc-again")
    assert first.submits == 1
    assert [row["client_order_id"] for row in resumed.orders] == ["dc-book"]


def test_matching_snapshot_still_submits(tmp_path: Path) -> None:
    broker = _BookBroker()
    session = _open(tmp_path, broker)
    session.arm(actor="ada")
    result = _buy(session, client_order_id="dc-ok")
    assert result["ok"] is True
    assert broker.submits == 1
    assert session.status()["halt_reason"] == ""
