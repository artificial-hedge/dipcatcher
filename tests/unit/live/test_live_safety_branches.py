"""Live-trading safety branches: reject, recover, stale marks, kill, and limits at the desk."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import (
    AlpacaBroker,
    BrokerAccount,
    BrokerAck,
    LiveOrder,
    LoopbackBroker,
)
from quant_fund.live.config import LiveConfig
from quant_fund.live.desk import LiveDesk, LiveDeskError
from quant_fund.live.errors import BrokerRejected, LiveGateError
from quant_fund.live.gateway_arm import arm_session, read_arm
from quant_fund.live.session import open_session
from quant_fund.pretrade.codes import INTERNAL
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)
_DIGEST = "a" * 64


def _desk_config(**overrides: object) -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(**overrides),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


def _desk(tmp_path: Path) -> LiveDesk:
    root = tmp_path / "desk"
    persisted = (root / "state.json").exists()
    cash = None if persisted else 1_000_000.0
    account = None if persisted else 1_000_000.0
    return LiveDesk.open(
        root,
        hmac_key=HMAC_KEY,
        config=_desk_config(),
        cash=cash,
        account_value=account,
        ts_ns=TS_NS,
    )


def _session_config(tmp: Path, *, venue: str = "loopback") -> LiveConfig:
    return LiveConfig.model_validate(
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


class _ScriptedBroker:
    """In-process broker double. It never opens a socket."""

    def __init__(
        self,
        *,
        submit_exc: BaseException | None = None,
        recovered: BrokerAck | None = None,
        cancel_exc: BaseException | None = None,
    ) -> None:
        self.submit_calls = 0
        self.recover_calls = 0
        self.cancel_calls = 0
        self.submit_exc = submit_exc
        self.recovered = recovered
        self.cancel_exc = cancel_exc
        self.marks: dict[str, float] = {}

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        return BrokerAccount(
            cash=1_000_000.0,
            equity=1_000_000.0,
            positions={},
            marks=dict(self.marks),
        )

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.submit_calls += 1
        if self.submit_exc is not None:
            raise self.submit_exc
        return BrokerAck(
            client_order_id=order.client_order_id,
            broker_order_id="scripted-1",
            status="accepted",
            filled_quantity=0.0,
            fill_price=None,
            fills=(),
            reason="",
        )

    def recover(self, client_order_id: str) -> BrokerAck | None:
        del client_order_id
        self.recover_calls += 1
        return self.recovered

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        del client_order_id, broker_order_id
        self.cancel_calls += 1
        if self.cancel_exc is not None:
            raise self.cancel_exc
        raise AssertionError("cancel was not expected")


def _armed_session(tmp: Path, broker: _ScriptedBroker):
    session = open_session(
        _session_config(tmp, venue="alpaca-paper"),
        PRETRADE,
        state_dir=tmp,
        now_ns=lambda: TS,
        broker=broker,
    )
    session.arm(actor="ada")
    return session


def _buy(session, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1.0,
        "limit_price": 100.0,
        "mark": 100.0,
        "actor": "ada",
        "at_ns": TS,
        "client_order_id": "dc-safety-1",
    }
    payload.update(overrides)
    return session.submit(**payload)


def test_loopback_rejects_cash_position_and_side_without_raising() -> None:
    broker = LoopbackBroker(cash=100.0, positions={"AAPL": 1.0})
    sell = LiveOrder("c-sell", "AAPL", "sell", 2.0, 10.0, 1)
    rejected = broker.submit(sell)
    assert rejected.status == "rejected"
    assert rejected.reason == "insufficient position"
    assert broker.positions["AAPL"] == 1.0
    assert broker.cash == 100.0

    buy = LiveOrder("c-buy", "AAPL", "buy", 10.0, 50.0, 2)
    cash = broker.submit(buy)
    assert cash.status == "rejected"
    assert cash.reason == "insufficient cash"
    assert broker.cash == 100.0

    again = broker.submit(sell)
    assert again == rejected
    assert broker.recover("missing") is None
    assert broker.recover("c-sell") == rejected

    with pytest.raises(BrokerRejected, match="buy or sell"):
        broker.submit(LiveOrder("c-bad", "AAPL", "short", 1.0, 10.0, 3))
    with pytest.raises(BrokerRejected, match="not at the loopback"):
        broker.cancel("missing", "")
    assert broker.cancel("c-sell", rejected.broker_order_id) == rejected


def test_loopback_cancel_of_a_deferred_accept_does_not_fill() -> None:
    broker = LoopbackBroker(cash=1_000.0, positions={})
    broker.defer_fill = True
    order = LiveOrder("c-work", "MSFT", "buy", 1.0, 20.0, 4)
    accepted = broker.submit(order)
    assert accepted.status == "accepted"
    assert broker.positions == {}
    canceled = broker.cancel(order.client_order_id, accepted.broker_order_id)
    assert canceled.status == "canceled"
    assert broker.recover(order.client_order_id) == canceled
    assert broker.cash == 1_000.0


def test_broker_rejection_is_an_ack_and_a_lost_response_can_be_recovered(tmp_path: Path) -> None:
    rejected = _ScriptedBroker(submit_exc=BrokerRejected("no buying power at the venue"))
    session = _armed_session(tmp_path / "reject", rejected)
    result = _buy(session)
    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert result["reason"] == "no buying power at the venue"
    assert result["live_pnl_claim"] is False
    # A definite venue rejection is still confirmed with one lookup. Not-found
    # is what lets the ack stay rejected.
    assert rejected.recover_calls == 1
    replay = _buy(session)
    assert replay["replayed"] is True
    assert rejected.submit_calls == 1

    recovered = BrokerAck(
        client_order_id="dc-recover-1",
        broker_order_id="broker-found",
        status="accepted",
        filled_quantity=0.0,
        fill_price=None,
        fills=(),
        reason="",
    )
    lost = _ScriptedBroker(submit_exc=TimeoutError("response lost"), recovered=recovered)
    resumed = _armed_session(tmp_path / "recover", lost)
    found = _buy(resumed, client_order_id="dc-recover-1")
    assert found["ok"] is True
    assert found["status"] == "accepted"
    assert found["broker_order_id"] == "broker-found"
    assert lost.recover_calls == 1
    assert lost.submit_calls == 1


def test_gate_error_during_submit_stays_not_acknowledged(tmp_path: Path) -> None:
    broker = _ScriptedBroker(submit_exc=LiveGateError("transport refused the host"))
    session = _armed_session(tmp_path, broker)
    with pytest.raises(LiveGateError, match="transport refused the host"):
        _buy(session, client_order_id="dc-gate-1")
    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert stored["orders"][0]["status"] == "not_acknowledged"
    assert stored["orders"][0]["reason"] == "broker credentials are not set"
    replay = _buy(session, client_order_id="dc-gate-1")
    assert replay["replayed"] is True
    assert replay["status"] == "not_acknowledged"
    assert broker.submit_calls == 1


def test_cancel_rejection_does_not_mark_the_order_canceled(tmp_path: Path) -> None:
    broker = _ScriptedBroker(cancel_exc=BrokerRejected("venue rejected the cancel"))
    session = _armed_session(tmp_path, broker)
    accepted = _buy(session, client_order_id="dc-cancel-1")
    assert accepted["status"] == "accepted"
    canceled = session.cancel(client_order_id="dc-cancel-1", actor="ada", at_ns=TS + 1)
    assert canceled["status"] == "accepted"
    assert canceled["reason"] == "venue rejected the cancel"
    assert broker.cancel_calls == 1
    assert session.orders[0]["status"] == "accepted"
    with pytest.raises(LiveGateError, match="not in this session"):
        session.cancel(client_order_id="missing-order", actor="ada", at_ns=TS + 2)


def test_alpaca_duplicate_client_id_recovers_and_other_rejects_do_not() -> None:
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):
        del headers, body, timeout
        calls.append(f"{method} {url}")
        if method == "POST":
            return 422, json.dumps({"message": "client_order_id must be unique"}).encode()
        return 200, json.dumps(
            {
                "id": "ord-1",
                "client_order_id": "client-1",
                "status": "new",
                "filled_qty": "0",
                "symbol": "SYNTH",
                "side": "buy",
            }
        ).encode()

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    order = LiveOrder("client-1", "SYNTH", "buy", 1.0, 10.0, 1)
    ack = broker.submit(order)
    assert ack.status == "accepted"
    assert ack.broker_order_id == "ord-1"
    assert any("orders:by_client_order_id" in call for call in calls)

    def reject_without_lookup(method, url, headers, body, timeout):
        del url, headers, body, timeout
        assert method == "POST"
        return 422, json.dumps({"message": "insufficient buying power"}).encode()

    plain = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=reject_without_lookup)
    with pytest.raises(BrokerRejected, match="insufficient buying power"):
        plain.submit(order)

    def missing_after_duplicate(method, url, headers, body, timeout):
        del url, headers, body, timeout
        if method == "POST":
            return 422, json.dumps({"message": "client_order_id must be unique"}).encode()
        return 404, b""

    lost = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=missing_after_duplicate)
    with pytest.raises(BrokerRejected, match="client_order_id"):
        lost.submit(order)


def test_alpaca_refuses_a_url_outside_the_venue_host_and_missing_credentials() -> None:
    seen: list[str] = []

    def transport(method, url, headers, body, timeout):
        del method, url, headers, body, timeout
        seen.append("called")
        return 200, b"{}"

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    with pytest.raises(LiveGateError, match="outside the venue host"):
        broker._request("GET", "not-a-path", None)
    assert seen == []

    bare = AlpacaBroker("alpaca-paper", None, transport=transport)
    with pytest.raises(LiveGateError, match="credentials are not set"):
        bare.submit(LiveOrder("client-2", "SYNTH", "buy", 1.0, 10.0, 1))
    assert seen == []


def test_stale_arm_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    arm_session(path, venue="loopback", actor="ada", pretrade_sha256=_DIGEST)
    armed = read_arm(path, venue="loopback", pretrade_sha256=_DIGEST)
    assert armed is not None
    assert armed["actor"] == "ada"
    with pytest.raises(LiveGateError, match="stale for the current pre-trade config"):
        read_arm(path, venue="loopback", pretrade_sha256="b" * 64)


def test_desk_flatten_skips_stale_marks_and_kill_blocks_the_reduce(tmp_path: Path) -> None:
    with _desk(tmp_path) as desk:
        desk.arm("loopback", "", TS_NS)
        bought = desk.submit("aapl", "BUY", 2, 50.0, TS_NS + 1, client_order_id="buy-aapl")
        assert bought["status"] == "filled"
        desk.positions["DUST"] = 1e-12
        desk.marks["AAPL"] = 0.0
        stale = desk.flatten(TS_NS + 2)
        reports = stale["flatten"]
        assert isinstance(reports, list)
        assert reports[0]["reasons"] == ["stale_data"]  # type: ignore[index]
        assert all(not (isinstance(row, dict) and row.get("symbol") == "DUST") for row in reports)
        assert desk.positions["AAPL"] == 2.0

        desk.marks.pop("AAPL")
        missing = desk.flatten(TS_NS + 3)
        missing_rows = missing["flatten"]
        assert isinstance(missing_rows, list)
        assert missing_rows[0]["reasons"] == ["stale_data"]  # type: ignore[index]

        desk.mark("AAPL", 50.0, TS_NS + 4)
        reduced = desk.flatten(TS_NS + 5)
        reduced_rows = reduced["flatten"]
        assert isinstance(reduced_rows, list)
        assert reduced_rows[0]["status"] == "filled"  # type: ignore[index]
        assert desk.positions.get("AAPL", 0.0) == pytest.approx(0.0)

        later = TS_NS + 2_000_000_000
        desk.submit("aapl", "BUY", 2, 50.0, later, client_order_id="buy-again")
        desk.kill("ops", "halt before reduce", later + 1)
        halted = desk.flatten(later + 2)
        halted_rows = halted["flatten"]
        assert isinstance(halted_rows, list)
        assert halted_rows[0]["accepted"] is False  # type: ignore[index]
        assert desk.positions["AAPL"] == 2.0
        assert halted["live_pnl_claim"] is False


def test_desk_reset_refuses_integrity_and_internal_faults(tmp_path: Path) -> None:
    with _desk(tmp_path) as desk:
        with pytest.raises(LiveDeskError, match="unknown venue"):
            desk.arm("alpaca-live", "", TS_NS)
        with pytest.raises(LiveDeskError, match="non-negative"):
            desk.kill("ops", "bad clock", -1)
        desk.kill("ops", "manual halt", TS_NS + 1)
        cleared = desk.reset("ops", "reviewed", TS_NS + 2)
        assert cleared["status"] == "reset"
        assert cleared["kill_latched"] is False
        assert cleared["transmission_after_reset"] is False
        released = desk.submit("aapl", "BUY", 1, 10.0, TS_NS + 3, client_order_id="after-reset")
        assert released["accepted"] is True
        desk.kill("ops", "manual halt", TS_NS + 4)
        desk.engine.book.kill_reason |= INTERNAL
        with pytest.raises(LiveDeskError, match="internal fault"):
            desk.reset("ops", "reviewed again", TS_NS + 5)
        denied = desk.submit("aapl", "BUY", 1, 10.0, TS_NS + 6, client_order_id="still-halted")
        assert denied["accepted"] is False
        assert denied["kill_latched"] is True

    torn = tmp_path / "torn"
    with _desk(torn) as desk:
        desk.submit("aapl", "BUY", 1, 100.0, TS_NS, client_order_id="journal-row")
    journal = torn / "desk" / "journal.jsonl"
    text = journal.read_text(encoding="utf-8")
    journal.write_text(text.replace("shadowed", "filled"), encoding="utf-8")
    with _desk(torn) as desk:
        with pytest.raises(LiveDeskError, match="journal"):
            desk.reset("ops", "cannot clear a torn journal", TS_NS + 4)
        assert desk.status()["transmission"] is False
        assert desk.status()["kill_latched"] is True


def test_desk_cancel_unknown_and_poll_on_a_bad_bridge_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _desk(tmp_path) as desk:
        unknown = desk.cancel("missing-order", TS_NS)
        assert unknown["accepted"] is False
        assert unknown["status"] == "unknown"
        shadowed = desk.submit("aapl", "BUY", 1, 100.0, TS_NS + 1, client_order_id="shadow-1")
        order = shadowed["order"]
        assert isinstance(order, dict)
        still = desk.cancel(str(order["order_id"]), TS_NS + 2)
        assert still["status"] == "unknown"

    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    with _desk(tmp_path / "drop") as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 1, 100.0, TS_NS + 1, client_order_id="work-1")
        order = sent["order"]
        assert isinstance(order, dict)
        backwards = desk.cancel(str(order["order_id"]), TS_NS)
        assert backwards["accepted"] is False
        assert backwards["reasons"] == ["non_monotonic_time"]
        inbound = tmp_path / "drop" / "desk" / "inbound"
        inbound.mkdir()
        (inbound / "bad.json").write_text("{", encoding="utf-8")
        polled = desk.poll(TS_NS + 2)
        assert polled["accepted"] is False
        assert "invalid JSON" in str(polled["error"])
        with pytest.raises(LiveDeskError, match="invalid JSON"):
            desk.reset("ops", "bridge froze the desk", TS_NS + 3)
        assert desk.status()["kill_latched"] is True


def test_session_flatten_skips_a_position_with_no_mark(tmp_path: Path) -> None:
    broker = LoopbackBroker(cash=1_000_000.0)
    session = open_session(
        _session_config(tmp_path),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=lambda: TS,
        broker=broker,
    )
    session.arm(actor="ada")
    broker.positions["MSFT"] = 2.0
    broker.positions["DUST"] = 1e-12
    flat = session.flatten(actor="ada", at_ns=TS)
    assert flat["ok"] is False
    assert flat["live_pnl_claim"] is False
    rows = flat["orders"]
    assert isinstance(rows, list)
    assert rows == [
        {
            "ok": False,
            "status": "skipped",
            "symbol": "MSFT",
            "reason": "no_mark",
            "live_pnl_claim": False,
            "research_evidence": False,
        }
    ]
    assert broker.positions["MSFT"] == 2.0
    assert broker.positions["DUST"] == 1e-12
