"""Crash cuts on the live journal. SYNTHETIC books, no sockets.

A torn tail is dropped. A corrupt earlier line fails closed. Cash and
positions after restart match the durable book, with no second fill.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
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

from quant_fund.execution.order_recon import Fill, Order
from quant_fund.live.brokers import LoopbackBroker
from quant_fund.live.desk import LiveDeskError
from quant_fund.live.errors import LiveGateError
from quant_fund.paper.broker_adapter import PaperBrokerAdapter

T0 = datetime(2024, 1, 3, 20, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", lambda _delay: None)


def _filled_loopback(root: Path) -> tuple[float, dict[str, float]]:
    broker = LoopbackBroker(cash=STARTING_CASH)
    session = open_fault_session(root, broker, venue="loopback")
    arm(session)
    filled = buy(session, quantity=10.0, client_order_id="gold")
    assert filled["status"] == "filled"
    return session.ledger.computed_cash, dict(session.ledger.computed_positions)


def test_torn_audit_tail_reopens_to_the_same_book(tmp_path: Path) -> None:
    cash, positions = _filled_loopback(tmp_path / "torn")
    audit = tmp_path / "torn" / "audit.jsonl"
    audit.write_bytes(audit.read_bytes() + b'{"event":"submit","brok')
    fresh = LoopbackBroker(cash=0.0)
    resumed = open_fault_session(tmp_path / "torn", fresh, venue="loopback")
    assert resumed.ledger.computed_cash == pytest.approx(cash)
    assert fresh.positions["AAPL"] == pytest.approx(positions["AAPL"])
    assert fresh.cash == pytest.approx(cash)
    report = resumed.reconcile(at_ns=TS + 1)
    assert report["ok"] is True
    assert report["duplicate_fills"] == 0
    assert resumed.ledger.computed_cash == pytest.approx(cash)
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    again = resumed.reconcile(at_ns=TS + 2)
    assert again["duplicate_fills"] == 0
    assert resumed.ledger.computed_cash == pytest.approx(cash)
    assert report["live_pnl_claim"] is False


def test_corrupt_audit_line_does_not_rewrite_positions(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    cash, positions = _filled_loopback(root)
    state = (root / "state.json").read_bytes()
    audit = root / "audit.jsonl"
    lines = audit.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    lines[0] = lines[0].replace('"event"', '"evnt"', 1)
    audit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LiveGateError, match="audit log"):
        open_fault_session(root, LoopbackBroker(cash=0.0), venue="loopback")
    assert (root / "state.json").read_bytes() == state
    stored = json.loads(state)
    assert stored["baseline_cash"] == pytest.approx(STARTING_CASH)
    assert cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert positions["AAPL"] == pytest.approx(10.0)


def test_crash_before_the_broker_leaves_no_fill(tmp_path: Path) -> None:
    broker = FaultBroker()
    root = tmp_path / "pre"
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)

    def _die(order: object) -> object:
        del order
        raise SystemExit("killed")

    session.broker.submit = _die  # type: ignore[method-assign]
    with pytest.raises(SystemExit, match="killed"):
        buy(session, quantity=10.0, client_order_id="pre-1")
    assert broker.submit_count == 0
    resumed = open_fault_session(root, broker, venue="alpaca-paper")
    assert resumed.orders[0]["status"] == "inflight"
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH)
    assert resumed.ledger.computed_positions == {}
    assert broker.positions == {}


def test_crash_after_accept_before_ledger_does_not_double_submit(tmp_path: Path) -> None:
    broker = FaultBroker()
    root = tmp_path / "ack"
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)
    real = session._apply_ack

    def _die(row: dict[str, object], ack: object, ts: int) -> None:
        real(row, ack, ts)  # type: ignore[arg-type]
        raise SystemExit("killed")

    session._apply_ack = _die  # type: ignore[method-assign]
    with pytest.raises(SystemExit, match="killed"):
        buy(session, quantity=4.0, client_order_id="ack-1")
    assert broker.submit_count == 1
    resumed = open_fault_session(root, broker, venue="alpaca-paper")
    assert resumed.orders[0]["status"] == "accepted"
    assert broker.submit_count == 1
    replay = buy(resumed, quantity=4.0, client_order_id="ack-1", at_ns=TS + 1)
    assert replay["replayed"] is True
    assert broker.submit_count == 1
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH)
    assert resumed.ledger.computed_positions == {}


def test_crash_after_broker_fill_before_ledger_adopts_once(tmp_path: Path) -> None:
    """The ack carries no print. The fills page does. Restart must apply it once."""
    broker = _PageOnly()
    root = tmp_path / "fill"
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)
    broker.drop_ack = True
    broker.drop_recover = True
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=10.0, client_order_id="fill-1")
    assert broker.deliver("fill-1", 10.0, 100.0, TS, "fill-1") == "applied"
    assert session.ledger.computed_positions == {}
    broker.drop_ack = False
    broker.drop_recover = False
    resumed = open_fault_session(root, broker, venue="alpaca-paper")
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert broker.cash == pytest.approx(resumed.ledger.computed_cash)
    assert broker.submit_count == 1
    again = open_fault_session(root, broker, venue="alpaca-paper")
    assert again.ledger.computed_cash == pytest.approx(resumed.ledger.computed_cash)
    assert again.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert broker.submit_count == 1


def test_crash_during_audit_append_keeps_the_persisted_fill(tmp_path: Path) -> None:
    broker = LoopbackBroker(cash=STARTING_CASH)
    root = tmp_path / "audit"
    session = open_fault_session(root, broker, venue="loopback")
    arm(session)
    real = session._audit

    def _die(event: str, fields: dict[str, object]) -> None:
        if event == "submit":
            with session.audit_path.open("ab") as handle:
                handle.write(b'{"event":"sub')
            raise SystemExit("killed")
        real(event, fields)

    session._audit = _die  # type: ignore[method-assign]
    with pytest.raises(SystemExit, match="killed"):
        buy(session, quantity=10.0, client_order_id="aud-1")
    cash = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert cash["orders"][0]["status"] == "filled"
    fresh = LoopbackBroker(cash=0.0)
    resumed = open_fault_session(root, fresh, venue="loopback")
    assert fresh.cash == pytest.approx(STARTING_CASH - 1_000.0)
    assert fresh.positions["AAPL"] == pytest.approx(10.0)
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    report = resumed.reconcile(at_ns=TS + 1)
    assert report["duplicate_fills"] == 0
    assert resumed.ledger.computed_cash == pytest.approx(STARTING_CASH - 1_000.0)


def test_paper_crash_before_replace_replays_the_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    def _die(src: object, dst: object) -> None:
        del src, dst
        raise SystemExit("killed")

    monkeypatch.setattr("os.replace", _die)
    with pytest.raises(SystemExit, match="killed"):
        run.checkpoint()
    monkeypatch.undo()
    resumed, report = PaperBrokerAdapter.resume(path, endpoint_kind="paper")
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    assert resumed.cash == pytest.approx(1_000.0)
    assert resumed.position("SEC_A") == pytest.approx(0.0)
    assert resumed.apply_fill(fill) == "accepted"
    assert resumed.cash == pytest.approx(900.0)
    assert resumed.position("SEC_A") == pytest.approx(10.0)
    assert resumed.apply_fill(fill) == "duplicate_fill"
    assert resumed.cash == pytest.approx(900.0)
    torn = path.read_bytes()[:12]
    path.write_bytes(torn)
    with pytest.raises((json.JSONDecodeError, Exception)):
        PaperBrokerAdapter.resume(path, endpoint_kind="paper")


def test_desk_torn_tail_then_inbound_fill_applies_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    root = tmp_path / "desk-torn"
    with open_desk(root) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 4, 100.0, TS_NS + 1, client_order_id="torn")
        order = sent["order"]
        assert isinstance(order, dict)
        order_id = str(order["order_id"])
        cash_before = float(desk.cash)
    journal = root / "desk" / "journal.jsonl"
    journal.write_bytes(journal.read_bytes() + b'{"kind":"fil')
    _write_fill(root / "desk" / "inbound", order_id, "fill-1", 4, 100)
    with open_desk(root) as desk:
        assert desk.cash == pytest.approx(cash_before)
        assert desk.positions == {}
        assert desk.halt_reason == ""
        report = desk.poll(TS_NS + 2)
    assert report["positions"] == {"AAPL": 4}
    assert report["cash"] == pytest.approx(STARTING_CASH - 400.0)
    with open_desk(root) as desk:
        again = desk.poll(TS_NS + 3)
    assert again["positions"] == {"AAPL": 4}
    assert again["cash"] == pytest.approx(STARTING_CASH - 400.0)
    assert again["live_pnl_claim"] is False


def test_desk_corrupt_journal_line_freezes_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    root = tmp_path / "desk-bad"
    with open_desk(root) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 4, 100.0, TS_NS + 1, client_order_id="bad")
        order = sent["order"]
        assert isinstance(order, dict)
        order_id = str(order["order_id"])
    _write_fill(root / "desk" / "inbound", order_id, "fill-1", 4, 100)
    with open_desk(root) as desk:
        applied = desk.poll(TS_NS + 2)
    assert applied["cash"] == pytest.approx(STARTING_CASH - 400.0)
    journal = root / "desk" / "journal.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"kind"', '"knd"', 1)
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open_desk(root) as desk:
        assert desk.cash == pytest.approx(STARTING_CASH - 400.0)
        assert desk.positions["AAPL"] == pytest.approx(4.0)
        assert "hash" in desk.halt_reason or "break" in desk.halt_reason
        report = desk.poll(TS_NS + 3)
        assert report["cash"] == pytest.approx(STARTING_CASH - 400.0)
        assert report["positions"]["AAPL"] == pytest.approx(4.0)
        with pytest.raises(LiveDeskError):
            desk.submit("aapl", "BUY", 1, 100.0, TS_NS + 4, client_order_id="after-freeze")
        assert desk.cash == pytest.approx(STARTING_CASH - 400.0)
        assert desk.positions["AAPL"] == pytest.approx(4.0)


def test_desk_crash_mid_fill_journal_replays_the_inbound_fill_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    root = tmp_path / "desk-cut"
    with open_desk(root) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 4, 100.0, TS_NS + 1, client_order_id="cut")
        order = sent["order"]
        assert isinstance(order, dict)
        order_id = str(order["order_id"])
    _write_fill(root / "desk" / "inbound", order_id, "fill-1", 4, 100)

    def _torn(path: Path, row: object) -> None:
        del row
        with path.open("ab") as handle:
            handle.write(b'{"kind":"fill"')
        raise SystemExit("killed")

    monkeypatch.setattr("quant_fund.live.desk._append_line", _torn)
    with open_desk(root) as desk, pytest.raises(SystemExit, match="killed"):
        desk.poll(TS_NS + 2)
    monkeypatch.undo()
    with open_desk(root) as desk:
        assert desk.positions == {}
        assert desk.cash == pytest.approx(STARTING_CASH)
        report = desk.poll(TS_NS + 3)
    assert report["positions"] == {"AAPL": 4}
    assert report["cash"] == pytest.approx(STARTING_CASH - 400.0)
    with open_desk(root) as desk:
        again = desk.poll(TS_NS + 4)
    assert again["cash"] == pytest.approx(STARTING_CASH - 400.0)
    assert again["positions"] == {"AAPL": 4}


class _PageOnly(FaultBroker):
    """Recover returns the id with no prints. Fills stay on the page."""

    def recover(self, client_order_id: str):  # type: ignore[no-untyped-def]
        from quant_fund.live.brokers import BrokerAck

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


def _write_fill(inbound: Path, order_id: str, event_id: str, qty: float, price: float) -> None:
    inbound.mkdir(parents=True, exist_ok=True)
    (inbound / f"{event_id}.json").write_text(
        json.dumps(
            {
                "schema": "live_broker_event.v1",
                "kind": "fill",
                "order_id": order_id,
                "event_id": event_id,
                "qty": qty,
                "price": price,
            }
        ),
        encoding="utf-8",
    )
