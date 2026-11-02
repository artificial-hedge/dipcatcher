"""Fail-closed checks for the armed gateway, desk, and broker adapter.

SYNTHETIC fills and marks are correctness fixtures, never market evidence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from quant_fund.live.arming import LIVE_CAPITAL_PHRASE
from quant_fund.live.bridge import BridgeError, DropBridge
from quant_fund.live.brokers import AlpacaBroker, BrokerAck, LiveOrder, LoopbackBroker
from quant_fund.live.config import LiveConfig, load_live_config
from quant_fund.live.desk import LiveDesk, LiveDeskError
from quant_fund.live.errors import BrokerRejected, BrokerTimeout, LiveBrokerRefused, LiveGateError
from quant_fund.live.ledger import LedgerFill, LiveLedger
from quant_fund.live.session import open_session
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from tests.unit.live.test_live_gateway import PRETRADE, TS, _arm, _buy, _clock, _config
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

_Transport = Any


def _desk_config(**overrides: object) -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(**overrides),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


def _desk(tmp_path: Path, **overrides: object) -> LiveDesk:
    root = tmp_path / "desk"
    return LiveDesk.open(
        root,
        hmac_key=HMAC_KEY,
        config=_desk_config(**overrides),
        cash=1_000_000.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )


def _recording_transport(calls: list[str]) -> _Transport:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        calls.append(url)
        return 200, b"{}"

    return transport


class _TimeoutCancel(LoopbackBroker):
    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        del client_order_id, broker_order_id
        raise TimeoutError("slow")


class _Overfill(LoopbackBroker):
    def submit(self, order: LiveOrder) -> BrokerAck:
        return BrokerAck(
            client_order_id=order.client_order_id,
            broker_order_id="over",
            status="filled",
            filled_quantity=order.quantity + 1.0,
            fill_price=order.limit_price,
            fills=(),
            reason="",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
        )


def test_live_config_missing_empty_and_invalid_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(LiveGateError, match="live config is missing"):
        load_live_config(tmp_path / "absent.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(LiveGateError, match="YAML mapping"):
        load_live_config(empty)
    broken = tmp_path / "broken.yaml"
    broken.write_text(":\n  - [", encoding="utf-8")
    with pytest.raises(LiveGateError, match="live config is invalid") as caught:
        load_live_config(broken)
    assert ":\n" not in str(caught.value)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: 2\nvenue: loopback\n", encoding="utf-8")
    with pytest.raises(LiveGateError, match="live config is invalid"):
        load_live_config(invalid)


def test_missing_pretrade_file_refuses_before_state_is_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing-pretrade.yaml"
    config = _config(tmp_path, pretrade_config=str(missing))
    state = tmp_path / "state"
    with pytest.raises(LiveGateError, match="pretrade config is missing"):
        open_session(config, missing, state_dir=state, now_ns=_clock)
    assert not state.exists()


def test_paper_config_refuses_a_live_broker_before_transport(tmp_path: Path) -> None:
    calls: list[str] = []
    broker = AlpacaBroker(
        "alpaca-live",
        ("test-key", "test-secret"),
        transport=_recording_transport(calls),
    )
    with pytest.raises(LiveGateError, match="broker venue does not match"):
        open_session(
            _config(tmp_path, venue="alpaca-paper"),
            PRETRADE,
            state_dir=tmp_path / "state",
            now_ns=_clock,
            broker=broker,
        )
    assert calls == []


def test_armed_alpaca_live_does_not_call_transport(tmp_path: Path) -> None:
    calls: list[str] = []
    broker = AlpacaBroker(
        "alpaca-live",
        ("test-key", "test-secret"),
        transport=_recording_transport(calls),
    )
    session = open_session(
        _config(tmp_path, venue="alpaca-live"),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=_clock,
        broker=broker,
    )
    session.arm(actor="ada", confirm_live_capital=LIVE_CAPITAL_PHRASE)
    with pytest.raises(LiveBrokerRefused, match="honesty contract"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            limit_price=100.0,
            mark=100.0,
            actor="ada",
        )
    assert calls == []


def test_kill_switch_denies_before_the_broker_is_called(tmp_path: Path) -> None:
    session = open_session(_config(tmp_path), PRETRADE, state_dir=tmp_path, now_ns=_clock)
    _arm(session)
    assert isinstance(session.broker, LoopbackBroker)
    session.kill(actor="ada", reason="operator halt", at_ns=TS)
    blocked = _buy(session, client_order_id="dc-blocked")
    assert blocked["ok"] is False
    assert blocked["broker_called"] is False
    assert "kill_switch" in blocked["reasons"]
    assert "dc-blocked" not in session.broker._orders


def test_client_order_id_conflict_does_not_replace_the_original(tmp_path: Path) -> None:
    session = open_session(_config(tmp_path), PRETRADE, state_dir=tmp_path, now_ns=_clock)
    _arm(session)
    filled = _buy(session, quantity=4.0, client_order_id="dc-same")
    assert filled["status"] == "filled"
    with pytest.raises(LiveGateError, match="conflicts with a different order"):
        _buy(session, quantity=9.0, client_order_id="dc-same")
    assert isinstance(session.broker, LoopbackBroker)
    assert session.broker.positions["AAPL"] == pytest.approx(4.0)
    replay = _buy(session, quantity=4.0, client_order_id="dc-same")
    assert replay["replayed"] is True
    assert session.broker.positions["AAPL"] == pytest.approx(4.0)


def test_cumulative_provisional_fill_replaces_instead_of_stacking() -> None:
    ledger = LiveLedger(baseline_cash=1_000.0)
    ledger.note_order("c1", expects_fill=True)
    first = LedgerFill("order:b1:cum", "c1", "AAPL", "buy", 1.0, 10.0, 20)
    assert ledger.record_fill(first) == "accepted"
    larger = LedgerFill("order:b1:cum", "c1", "AAPL", "buy", 3.0, 10.0, 30)
    assert ledger.record_fill(larger) == "replaced_provisional"
    assert ledger.computed_cash == pytest.approx(970.0)
    assert ledger.computed_positions["AAPL"] == pytest.approx(3.0)
    assert ledger.record_fill(larger) == "duplicate_fill"
    assert ledger.computed_cash == pytest.approx(970.0)
    smaller = LedgerFill("order:b1:cum", "c1", "AAPL", "buy", 2.0, 10.0, 40)
    with pytest.raises(ValueError, match="conflicts with a different execution"):
        ledger.record_fill(smaller)
    assert ledger.computed_cash == pytest.approx(970.0)
    assert ledger.computed_positions["AAPL"] == pytest.approx(3.0)


def test_orphan_provisional_growth_does_not_move_cash() -> None:
    ledger = LiveLedger(baseline_cash=1_000.0)
    first = LedgerFill("order:b1:cum", "missing", "AAPL", "buy", 1.0, 10.0, 20)
    assert ledger.record_fill(first) == "orphan_fill"
    larger = LedgerFill("order:b1:cum", "missing", "AAPL", "buy", 3.0, 10.0, 30)
    assert ledger.record_fill(larger) == "orphan_fill"
    assert ledger.computed_cash == pytest.approx(1_000.0)
    assert ledger.computed_positions.get("AAPL", 0.0) == pytest.approx(0.0)
    report = ledger.reconcile()
    assert report["orphan_fills"] == 1
    assert report["ok"] is False


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.nan, math.inf])
def test_alpaca_timeout_must_be_finite_and_positive(timeout: float) -> None:
    with pytest.raises(LiveGateError, match="timeout must be finite and positive"):
        AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), timeout=timeout)


def test_cancel_timeout_stays_unconfirmed(tmp_path: Path) -> None:
    broker = _TimeoutCancel(cash=1_000_000.0)
    broker.defer_fill = True
    session = open_session(
        _config(tmp_path),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=_clock,
        broker=broker,
    )
    _arm(session)
    working = _buy(session, client_order_id="dc-working")
    assert working["status"] == "accepted"
    with pytest.raises(LiveGateError, match="cancel unconfirmed after broker timeout"):
        session.cancel(client_order_id="dc-working", actor="ada", at_ns=TS + 1)
    assert session.orders[0]["status"] == "pending_cancel"
    assert session.orders[0]["status"] != "canceled"
    assert broker.positions == {}


def test_malformed_fill_row_is_rejected() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body, timeout
        return 200, b"[null]"

    broker = AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=transport)
    with pytest.raises(BrokerRejected, match="fill row was malformed"):
        broker.fills(broker_to_client={})


def test_overfill_ack_does_not_move_cash(tmp_path: Path) -> None:
    broker = _Overfill(cash=1_000_000.0)
    session = open_session(
        _config(tmp_path),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=_clock,
        broker=broker,
    )
    _arm(session)
    with pytest.raises(LiveGateError, match="exceeds the order quantity"):
        _buy(session, quantity=1.0, client_order_id="dc-over")
    assert session.ledger.computed_cash == pytest.approx(1_000_000.0)
    assert broker.cash == pytest.approx(1_000_000.0)


def test_flatten_refuses_a_stale_or_missing_mark(tmp_path: Path) -> None:
    session = open_session(_config(tmp_path), PRETRADE, state_dir=tmp_path, now_ns=_clock)
    _arm(session)
    _buy(session, quantity=4.0)
    assert isinstance(session.broker, LoopbackBroker)
    session.mark_ts["AAPL"] = TS - 6_000_000_000
    stale = session.flatten(actor="ada", at_ns=TS)
    assert stale["ok"] is False
    assert stale["orders"][0]["reason"] == "stale_data"
    assert session.broker.positions["AAPL"] == pytest.approx(4.0)

    bare = open_session(
        _config(tmp_path / "bare"),
        PRETRADE,
        state_dir=tmp_path / "bare",
        now_ns=_clock,
    )
    _arm(bare)
    assert isinstance(bare.broker, LoopbackBroker)
    bare.broker.positions["MSFT"] = 2.0
    missing = bare.flatten(actor="ada", at_ns=TS)
    assert missing["orders"][0]["reason"] == "no_mark"
    assert bare.orders == []
    assert bare.broker.positions["MSFT"] == pytest.approx(2.0)


def test_daily_loss_blocks_the_next_order(tmp_path: Path) -> None:
    session = open_session(_config(tmp_path), PRETRADE, state_dir=tmp_path, now_ns=_clock)
    _arm(session)
    bought = _buy(session, quantity=2_000.0, client_order_id="dc-loss")
    assert bought["status"] == "filled"
    denied = session.submit(
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        limit_price=50.0,
        mark=50.0,
        actor="ada",
        client_order_id="dc-again",
        at_ns=TS + 1,
    )
    assert denied["ok"] is False
    assert "daily_loss" in denied["reasons"]
    assert isinstance(session.broker, LoopbackBroker)
    assert session.broker.positions["AAPL"] == pytest.approx(2_000.0)


def test_desk_client_id_conflict_does_not_store_a_second_ticket(tmp_path: Path) -> None:
    with _desk(tmp_path) as desk:
        desk.submit("aapl", "BUY", 10, 100.0, TS_NS, client_order_id="c1")
        with pytest.raises(LiveDeskError, match="conflicts with a different order"):
            desk.submit("aapl", "BUY", 11, 100.0, TS_NS, client_order_id="c1")
        assert len(desk.orders) == 1


def test_desk_flatten_refuses_a_stale_mark(tmp_path: Path) -> None:
    with _desk(tmp_path, max_mark_age_ns=1) as desk:
        desk.positions["AAPL"] = 4.0
        desk.marks["AAPL"] = 100.0
        desk.mark_ts["AAPL"] = TS_NS
        report = desk.flatten(TS_NS + 5)
        assert report["flatten"][0]["reasons"] == ["stale_data"]
        assert desk.positions["AAPL"] == pytest.approx(4.0)
        assert len(desk.orders) == 0


def test_unreadable_drop_event_freezes_the_desk(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path / "drop")
    bridge.inbound.mkdir(parents=True)
    (bridge.inbound / "bad.json").mkdir()
    with pytest.raises(BridgeError, match="unreadable event"):
        bridge.poll()

    with _desk(tmp_path) as desk:
        inbound = tmp_path / "desk" / "inbound"
        inbound.mkdir()
        (inbound / "bad.json").mkdir()
        report = desk.poll(TS_NS)
        assert report["accepted"] is False
        assert "unreadable" in str(report["error"])
        assert desk.engine.book.killed == 1


def test_broker_timeout_is_not_a_rejection() -> None:
    assert not issubclass(BrokerTimeout, LiveGateError)


def test_live_config_model_rejects_a_blank_pretrade_path() -> None:
    with pytest.raises(ValueError):
        LiveConfig.model_validate(
            {
                "schema_version": 1,
                "venue": "loopback",
                "starting_cash": 1.0,
                "pretrade_config": "",
            }
        )
