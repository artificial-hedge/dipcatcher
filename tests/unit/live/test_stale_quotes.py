"""Live and paper-session orders stop when the quote cannot be used."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import BrokerAck, LiveOrder, LoopbackBroker
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


class _CountingBroker(LoopbackBroker):
    def __init__(self) -> None:
        super().__init__(cash=1_000_000.0)
        self.submits = 0

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.submits += 1
        return super().submit(order)


def _clock() -> int:
    return TS


def _config(tmp: Path) -> LiveConfig:
    return LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "loopback",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )


def _open(tmp: Path, broker: _CountingBroker | None = None) -> Any:
    return open_session(
        _config(tmp),
        PRETRADE,
        state_dir=tmp,
        now_ns=_clock,
        broker=broker or _CountingBroker(),
    )


def _arm(session: Any) -> None:
    session.arm(actor="ada")


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


def _desk(tmp: Path, *, max_age: int) -> LiveDesk:
    return LiveDesk.open(
        tmp / "desk",
        hmac_key=HMAC_KEY,
        config=PretradeConfig(
            schema_version=1,
            limits=limit_config(max_reference_age_ns=max_age, max_mark_age_ns=max_age),
            session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
        ),
        cash=1_000_000.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )


def test_plain_submit_still_sends_and_a_later_tick_inside_the_window_sends(tmp_path: Path) -> None:
    broker = _CountingBroker()
    session = _open(tmp_path, broker)
    _arm(session)
    first = _buy(session, client_order_id="fresh-1")
    assert first["ok"] is True
    age = int(session.engine._max_ref_age)
    second = _buy(
        session,
        client_order_id="fresh-2",
        at_ns=TS + age,
        quote_ts_ns=TS + age,
    )
    assert second["ok"] is True
    assert broker.submits == 2


def test_stale_explicit_quote_does_not_send(tmp_path: Path) -> None:
    broker = _CountingBroker()
    session = _open(tmp_path, broker)
    _arm(session)
    age = int(session.engine._max_ref_age)
    with pytest.raises(LiveGateError, match="stale quote for AAPL"):
        _buy(session, client_order_id="stale", quote_ts_ns=TS - (age + 1))
    assert broker.submits == 0
    assert session.orders == []
    assert broker.positions == {}


def test_clock_skew_and_duplicate_and_reorder_do_not_send(tmp_path: Path) -> None:
    broker = _CountingBroker()
    session = _open(tmp_path, broker)
    _arm(session)
    marked = session.mark(symbol="AAPL", price=100.0, actor="ada")
    assert marked["ok"] is True
    with pytest.raises(LiveGateError, match="duplicate quote for AAPL"):
        session.mark(symbol="AAPL", price=101.0, actor="ada")
    assert session.marks["AAPL"] == 100.0
    sent = _buy(session, client_order_id="ok")
    assert sent["ok"] is True
    assert broker.submits == 1
    with pytest.raises(LiveGateError, match="clock skew for AAPL"):
        _buy(session, client_order_id="ahead", at_ns=TS + 5, quote_ts_ns=TS + 6)
    with pytest.raises(LiveGateError, match="out-of-order quote for AAPL"):
        _buy(session, client_order_id="back", at_ns=TS + 5, quote_ts_ns=TS - 1)
    with pytest.raises(LiveGateError, match="duplicate quote for AAPL"):
        _buy(session, client_order_id="again", at_ns=TS + 5, quote_ts_ns=TS)
    assert broker.submits == 1
    assert [row["client_order_id"] for row in session.orders] == ["ok"]


def test_reopen_still_halts_a_quote_that_aged_out(tmp_path: Path) -> None:
    first = _open(tmp_path)
    _arm(first)
    age = int(first.engine._max_ref_age)
    assert _buy(first, client_order_id="kept", quote_ts_ns=TS)["ok"] is True
    second_broker = _CountingBroker()
    second = _open(tmp_path, second_broker)
    with pytest.raises(LiveGateError, match="stale quote for AAPL"):
        _buy(second, client_order_id="late", at_ns=TS + age + 1)
    assert second_broker.submits == 0
    assert [row["client_order_id"] for row in second.orders] == ["kept"]


def test_stored_bool_quote_clock_refuses_to_open(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    _buy(session, client_order_id="kept")
    path = tmp_path / "state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["quote_ns"] = {"AAPL": True}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LiveGateError, match="not usable"):
        _open(tmp_path)


def test_desk_duplicate_and_reordered_marks_do_not_create_orders(tmp_path: Path) -> None:
    desk = _desk(tmp_path, max_age=5_000_000_000)
    try:
        desk.mark("aapl", 100.0, TS_NS)
        with pytest.raises(LiveDeskError, match="duplicate quote for AAPL"):
            desk.mark("aapl", 100.0, TS_NS)
        with pytest.raises(LiveDeskError, match="out-of-order quote for AAPL"):
            desk.mark("aapl", 100.0, TS_NS - 1)
        assert desk.orders == {}
        assert desk.mark_ts["AAPL"] == TS_NS
    finally:
        desk.close()


def test_desk_stale_quote_does_not_transmit_and_a_new_tick_can(tmp_path: Path) -> None:
    age = 5_000_000_000
    desk = _desk(tmp_path, max_age=age)
    try:
        desk.arm("loopback", "", TS_NS)
        first = desk.submit("aapl", "BUY", 1, 100.0, TS_NS, client_order_id="fresh")
        assert first["transmitted"] is True
        with pytest.raises(LiveDeskError, match="stale quote for AAPL"):
            desk.submit("aapl", "BUY", 1, 100.0, TS_NS + age + 1, client_order_id="stale")
        clients = [order.client_order_id for order in desk.orders.values()]
        assert clients == ["fresh"]
        desk.mark("aapl", 100.0, TS_NS + age + 1)
        nxt = desk.submit("aapl", "BUY", 1, 100.0, TS_NS + age + 1, client_order_id="after")
        assert nxt["transmitted"] is True
        clients = [order.client_order_id for order in desk.orders.values()]
        assert "stale" not in clients
        assert "after" in clients
    finally:
        desk.close()


def test_desk_quote_ahead_of_the_order_clock_does_not_send(tmp_path: Path) -> None:
    desk = _desk(tmp_path, max_age=5_000_000_000)
    try:
        desk.arm("loopback", "", TS_NS)
        desk.mark("aapl", 100.0, TS_NS + 10)
        with pytest.raises(LiveDeskError, match="clock skew for AAPL"):
            desk.submit("aapl", "BUY", 1, 100.0, TS_NS, client_order_id="skew")
        assert desk.orders == {}
    finally:
        desk.close()
