"""Live-session acknowledgement and execution ordering edge cases."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import BrokerAccount, BrokerAck, BrokerFill, FillPage, LiveOrder
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import BrokerRejected, LiveGateError
from quant_fund.live.session import LiveSession, open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def _ack(
    client: str = "order-1",
    *,
    broker_id: str = "broker-1",
    status: str = "accepted",
    quantity: float = 0.0,
    price: float | None = None,
    fills: tuple[BrokerFill, ...] = (),
    reason: str = "",
) -> BrokerAck:
    return BrokerAck(client, broker_id, status, quantity, price, fills, reason)


def _fill(
    fill_id: str,
    quantity: float,
    price: float = 100.0,
    *,
    client: str = "order-1",
    at_ns: int = TS,
) -> BrokerFill:
    return BrokerFill(fill_id, client, "AAPL", "buy", quantity, price, at_ns)


class _ScriptedBroker:
    def __init__(self, submit_ack: BrokerAck) -> None:
        self.submit_ack = submit_ack
        self.submit_error: BrokerRejected | None = None
        self.cancel_ack: BrokerAck | None = None
        self.cancel_error: BrokerRejected | None = None
        self.recover_ack: BrokerAck | None = None
        self.recover_script: list[BrokerAck | None] = []
        self.recover_error: Exception | None = None
        self.page = FillPage((), False)
        self.marks: dict[str, float] = {}
        self.cash = 1_000_000.0
        self.positions: dict[str, float] = {}
        self._accounted: set[str] = set()
        self.submit_calls = 0
        self.cancel_calls = 0
        self.recover_calls = 0

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[symbol] = price

    def submit(self, order: LiveOrder) -> BrokerAck:
        del order
        self.submit_calls += 1
        if self.submit_error is not None:
            raise self.submit_error
        self.observe(self.submit_ack)
        return self.submit_ack

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        del client_order_id, broker_order_id
        self.cancel_calls += 1
        if self.cancel_error is not None:
            raise self.cancel_error
        assert self.cancel_ack is not None
        self.observe(self.cancel_ack)
        return self.cancel_ack

    def recover(self, client_order_id: str) -> BrokerAck | None:
        del client_order_id
        self.recover_calls += 1
        if self.recover_error is not None:
            raise self.recover_error
        recovered = self.recover_script.pop(0) if self.recover_script else self.recover_ack
        if recovered is not None:
            self.observe(recovered)
        return recovered

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        for fill in self.page.fills:
            self._account(fill)
        return self.page

    def account(self) -> BrokerAccount:
        equity = self.cash + sum(
            quantity * self.marks.get(symbol, 0.0) for symbol, quantity in self.positions.items()
        )
        return BrokerAccount(self.cash, equity, dict(self.positions), dict(self.marks))

    def observe(self, ack: BrokerAck) -> None:
        for fill in ack.fills:
            self._account(fill)

    def _account(self, fill: BrokerFill) -> None:
        if fill.fill_id in self._accounted:
            return
        self._accounted.add(fill.fill_id)
        signed = fill.quantity if fill.side == "buy" else -fill.quantity
        self.cash -= signed * fill.price
        self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) + signed


def _open(tmp_path: Path, broker: _ScriptedBroker) -> LiveSession:
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "alpaca-paper",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp_path),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    return open_session(config, PRETRADE, state_dir=tmp_path, now_ns=lambda: TS, broker=broker)


def _order(session: LiveSession, **overrides: object) -> dict[str, Any]:
    payload: dict[str, object] = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10.0,
        "limit_price": 100.0,
        "mark": 100.0,
        "actor": "test",
        "client_order_id": "order-1",
        "at_ns": TS,
    }
    payload.update(overrides)
    return session.submit(**payload)  # type: ignore[arg-type]


def _session(tmp_path: Path, broker: _ScriptedBroker, *, quantity: float = 10.0) -> LiveSession:
    session = _open(tmp_path, broker)
    session.arm(actor="test")
    result = _order(session, quantity=quantity)
    assert result["broker_called"] is True
    return session


def _apply(session: LiveSession, broker: _ScriptedBroker, ack: BrokerAck) -> None:
    broker.observe(ack)
    session._apply_ack(session.orders[0], ack, TS)


def test_partial_fill_ack_is_monotonic_and_duplicate_is_idempotent(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)
    row = session.orders[0]
    partial = _ack(
        status="partially_filled",
        quantity=4.0,
        price=100.0,
        fills=(_fill("fill-1", 4.0),),
    )

    _apply(session, broker, partial)
    _apply(session, broker, partial)
    _apply(session, broker, _ack(status="accepted", quantity=1.0, price=99.0))

    assert row["broker_order_id"] == "broker-1"
    assert row["status"] == "partially_filled"
    assert row["filled_quantity"] == pytest.approx(4.0)
    assert row["fill_price"] == pytest.approx(100.0)
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(4.0)
    assert session.ledger.computed_cash == pytest.approx(999_600.0)


@pytest.mark.parametrize("terminal", ["canceled", "rejected"])
def test_terminal_cancel_and_reject_do_not_reopen(tmp_path: Path, terminal: str) -> None:
    broker = _ScriptedBroker(_ack(status=terminal, reason=terminal))
    session = _session(tmp_path, broker)
    row = session.orders[0]

    _apply(session, broker, _ack(status="accepted", broker_id="broker-1"))

    assert row["status"] == terminal
    assert row["filled_quantity"] == 0.0
    assert row["reason"] == terminal


def test_filled_order_cannot_regress_to_a_stale_partial_ack(tmp_path: Path) -> None:
    fill = _fill("fill-1", 10.0)
    broker = _ScriptedBroker(_ack(status="filled", quantity=10.0, price=100.0, fills=(fill,)))
    session = _session(tmp_path, broker)
    row = session.orders[0]

    _apply(session, broker, _ack(status="partially_filled", quantity=3.0, price=99.0))

    assert row["status"] == "filled"
    assert row["filled_quantity"] == pytest.approx(10.0)
    assert row["fill_price"] == pytest.approx(100.0)
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("client", "broker_id"),
    [("other-order", "broker-1"), ("order-1", "other-broker")],
)
def test_ack_identity_mismatch_fails_before_order_or_accounting_mutation(
    tmp_path: Path, client: str, broker_id: str
) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)
    before = dict(session.orders[0])
    mismatched = _ack(
        client=client,
        broker_id=broker_id,
        status="filled",
        quantity=10.0,
        price=100.0,
        fills=(_fill("wrong-fill", 10.0, client=client),),
    )

    with pytest.raises(LiveGateError, match="ack.*identity"):
        _apply(session, broker, mismatched)

    assert session.orders[0] == before
    assert session.ledger.fills() == []
    assert session.ledger.computed_positions == {}


def test_cancel_ack_keeps_executions_and_late_fills_do_not_reopen(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    first_race_fill = _fill("fill-1", 4.0, 100.0, at_ns=TS + 1)
    broker.cancel_ack = _ack(
        status="canceled",
        quantity=4.0,
        price=100.0,
        fills=(first_race_fill,),
    )
    session = _session(tmp_path, broker)

    canceled = session.cancel(client_order_id="order-1", actor="test", at_ns=TS + 2)

    assert canceled["status"] == "canceled"
    assert canceled["filled_quantity"] == pytest.approx(4.0)
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(4.0)

    late_fill = _fill("fill-2", 6.0, 110.0, at_ns=TS + 3)
    broker.page = FillPage((first_race_fill, late_fill), False)
    report = session.reconcile(at_ns=TS + 4)

    assert report["ok"] is True
    row = session.orders[0]
    assert row["status"] == "canceled"
    assert row["filled_quantity"] == pytest.approx(10.0)
    assert row["fill_price"] == pytest.approx(106.0)
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert session.ledger.computed_cash == pytest.approx(998_940.0)
    replay = session.cancel(client_order_id="order-1", actor="test", at_ns=TS + 5)
    assert replay["replayed"] is True
    assert broker.cancel_calls == 1


def test_reconciled_overfill_fails_closed_without_dropping_accounting(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker, quantity=5.0)
    broker.page = FillPage((_fill("too-much", 6.0),), False)

    with pytest.raises(LiveGateError, match="fills exceed order quantity"):
        session.reconcile(at_ns=TS + 1)

    assert session.orders[0]["filled_quantity"] == 0.0
    assert session.orders[0]["status"] == "accepted"
    assert len(session.ledger.fills()) == 1
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(6.0)
    assert session.ledger.computed_cash == pytest.approx(999_400.0)


@pytest.mark.parametrize(("symbol", "side"), [("MSFT", "buy"), ("AAPL", "sell")])
def test_known_order_fill_identity_is_checked_before_accounting(
    tmp_path: Path, symbol: str, side: str
) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)
    mismatched = BrokerFill("bad-fill", "order-1", symbol, side, 1.0, 100.0, TS)

    with pytest.raises(LiveGateError, match="fill.*does not match"):
        session._record_fill(mismatched, TS)

    assert session.ledger.fills() == []
    assert session._model_settled == pytest.approx(1_000_000.0)


def test_ack_fill_identities_are_prevalidated_before_any_fill_is_applied(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)
    before = dict(session.orders[0])
    valid = _fill("valid-fill", 2.0)
    invalid = BrokerFill("invalid-fill", "order-1", "MSFT", "buy", 1.0, 100.0, TS)
    ack = _ack(
        status="partially_filled",
        quantity=3.0,
        price=100.0,
        fills=(valid, invalid),
    )

    with pytest.raises(LiveGateError, match="fill.*does not match"):
        _apply(session, broker, ack)

    assert session.orders[0] == before
    assert session.ledger.fills() == []
    assert session._model_settled == pytest.approx(1_000_000.0)


def test_conflicting_fill_does_not_mutate_risk_settlement(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)
    first = _fill("fill-1", 2.0)
    _apply(
        session,
        broker,
        _ack(status="partially_filled", quantity=2.0, price=100.0, fills=(first,)),
    )
    settled = session._model_settled
    engine_settled = session.engine.book.settled
    conflicting = _fill("fill-1", 3.0)

    assert session._record_fill(conflicting, TS) == "conflicting_fill"

    assert session._model_settled == pytest.approx(settled)
    assert session.engine.book.settled == pytest.approx(engine_settled)
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(2.0)


def test_two_partial_fills_update_settlement_and_survive_resume(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)
    first = _fill("fill-1", 2.0, 100.0, at_ns=TS + 1)
    second = _fill("fill-2", 3.0, 110.0, at_ns=TS + 2)
    _apply(
        session,
        broker,
        _ack(status="partially_filled", quantity=2.0, price=100.0, fills=(first,)),
    )
    _apply(
        session,
        broker,
        _ack(status="partially_filled", quantity=5.0, price=106.0, fills=(second,)),
    )

    assert session._model_settled == pytest.approx(999_470.0)
    assert session.engine.book.settled == pytest.approx(999_470.0)
    assert session.ledger.computed_cash == pytest.approx(999_470.0)

    session._persist()
    resumed = _open(tmp_path, broker)

    assert resumed._model_settled == pytest.approx(999_470.0)
    assert resumed.engine.book.settled == pytest.approx(999_470.0)
    assert resumed.ledger.computed_cash == pytest.approx(999_470.0)


def test_provisional_replacement_rebuilds_instead_of_double_counting(tmp_path: Path) -> None:
    cumulative = "order:broker-1:cum"
    initial = _fill(cumulative, 2.0, 100.0)
    broker = _ScriptedBroker(
        _ack(status="partially_filled", quantity=2.0, price=100.0, fills=(initial,))
    )
    session = _session(tmp_path, broker)
    updated = _fill(cumulative, 4.0, 105.0, at_ns=TS + 1)

    _apply(
        session,
        broker,
        _ack(status="partially_filled", quantity=4.0, price=105.0, fills=(updated,)),
    )

    assert session._model_settled == pytest.approx(999_580.0)
    real = _fill("venue-fill-1", 4.0, 105.0, at_ns=TS + 2)
    assert session._record_fill(real, TS + 2) == "replaced_provisional"
    assert session._model_settled == pytest.approx(999_580.0)
    assert session.engine.book.settled == pytest.approx(999_580.0)
    assert session.ledger.computed_cash == pytest.approx(999_580.0)
    settled = session._model_settled
    assert session._record_fill(real, TS + 2) == "duplicate_fill"
    assert session._model_settled == pytest.approx(settled)


def test_orphan_fill_is_not_replayed_into_risk_accounting(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)
    orphan = _fill("orphan-fill", 2.0, client="unknown-order")

    assert session._record_fill(orphan, TS) == "orphan_fill"
    session._persist()
    resumed = _open(tmp_path, broker)

    assert len(resumed.ledger.fills()) == 1
    assert resumed._model_settled == pytest.approx(1_000_000.0)
    assert resumed.engine.book.settled == pytest.approx(1_000_000.0)


def test_identical_client_order_reuse_is_replayed(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)

    replay = _order(session, mark=101.0)

    assert replay["replayed"] is True
    assert replay["status"] == "accepted"
    assert broker.submit_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("symbol", "MSFT"), ("side", "sell"), ("quantity", 11.0), ("limit_price", 101.0)],
)
def test_conflicting_client_order_reuse_is_denied_without_broker_calls(
    tmp_path: Path, field: str, value: object
) -> None:
    broker = _ScriptedBroker(_ack())
    session = _session(tmp_path, broker)
    calls = (broker.submit_calls, broker.cancel_calls, broker.recover_calls)

    denied = _order(session, **{field: value})

    assert denied["ok"] is False
    assert denied["status"] == "denied"
    assert denied["reasons"] == ["duplicate_order"]
    assert denied["broker_called"] is False
    assert (broker.submit_calls, broker.cancel_calls, broker.recover_calls) == calls


def test_submit_decode_rejection_uses_recovered_order(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack(status="rejected"))
    broker.submit_error = BrokerRejected("broker HTTP 200 returned non-JSON")
    broker.recover_ack = _ack(status="accepted")
    session = _open(tmp_path, broker)
    session.arm(actor="test")

    result = _order(session)

    assert result["status"] == "accepted"
    assert broker.submit_calls == 1
    assert broker.recover_calls == 1


def test_submit_rejection_with_failed_recovery_stays_recoverable(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    broker.submit_error = BrokerRejected("broker HTTP 200 returned non-JSON")
    broker.recover_error = TimeoutError("lookup failed")
    session = _open(tmp_path, broker)
    session.arm(actor="test")

    with pytest.raises(LiveGateError, match="stays inflight"):
        _order(session)

    assert session.orders[0]["status"] == "inflight"
    assert "recoverable" in session.orders[0]["reason"]


def test_genuine_submit_reject_with_successful_not_found_remains_rejected(
    tmp_path: Path,
) -> None:
    broker = _ScriptedBroker(_ack())
    broker.submit_error = BrokerRejected("order rejected by venue")
    session = _open(tmp_path, broker)
    session.arm(actor="test")

    result = _order(session)

    assert result["status"] == "rejected"
    assert result["reason"] == "order rejected by venue"
    assert broker.recover_calls == 1


def test_cancel_rejection_without_recovery_is_not_reported_as_success(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    broker.cancel_error = BrokerRejected("cancel response malformed")
    session = _session(tmp_path, broker)

    result = session.cancel(client_order_id="order-1", actor="test", at_ns=TS + 1)

    assert result["ok"] is False
    assert result["status"] == "accepted"
    assert result["reason"] == "cancel response malformed"


def test_cancel_rejection_applies_recovered_terminal_state(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    broker.cancel_error = BrokerRejected("cancel response malformed")
    broker.recover_script = [None, _ack(status="canceled")]
    session = _session(tmp_path, broker)

    result = session.cancel(client_order_id="order-1", actor="test", at_ns=TS + 1)

    assert result["ok"] is True
    assert result["status"] == "canceled"
    assert broker.cancel_calls == 1


def test_pending_cancel_is_recovered_before_retrying_cancel(tmp_path: Path) -> None:
    broker = _ScriptedBroker(_ack())
    broker.cancel_ack = _ack(status="pending_cancel")
    session = _session(tmp_path, broker)

    pending = session.cancel(client_order_id="order-1", actor="test", at_ns=TS + 1)
    assert pending["status"] == "pending_cancel"
    assert pending["ok"] is False

    broker.recover_ack = _ack(status="canceled")
    recovered = session.cancel(client_order_id="order-1", actor="test", at_ns=TS + 2)

    assert recovered["status"] == "canceled"
    assert recovered["replayed"] is True
    assert broker.cancel_calls == 1
