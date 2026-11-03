"""Order lifecycle: legal traces stay inside the spec, illegal ones do not apply.

Hypothesis is derandomized (fixed seed, no example database) and each test
draws about 20 examples. ``new`` is the acked working state (broker ``ACKED``).
From there a trace may rest as ``partial`` and end ``filled``, ``canceled``,
``rejected``, or ``expired``. A cancel after a full fill does not change the
fill. Loopback prints are SYNTHETIC and ``live_pnl_claim`` stays false.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.execution.order_recon import Fill, Order, OrderFillLedger
from quant_fund.formal.order_lifecycle import TERMINAL, Book, check_trace, step
from quant_fund.live.brokers import LiveOrder, LoopbackBroker
from quant_fund.live.desk import LiveDesk
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

_EPS = 1e-8
_FAST = settings(derandomize=True, max_examples=20, deadline=None)
_T0 = datetime(2024, 1, 2, tzinfo=UTC)
_CASH = 100_000.0
_TERMINAL_OP = {"cancel": "canceled", "reject": "rejected", "expire": "expired"}


def _desk_config() -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


def _open_desk(root: Path) -> LiveDesk:
    return LiveDesk.open(
        root / "desk",
        hmac_key=HMAC_KEY,
        config=_desk_config(),
        cash=1_000_000.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )


def _chunks(draw: st.DrawFn, total: int, parts: int) -> list[int]:
    left = total
    out: list[int] = []
    for index in range(parts - 1):
        hi = left - (parts - index - 1)
        take = draw(st.integers(1, hi))
        out.append(take)
        left -= take
    out.append(left)
    return out


@st.composite
def _legal_trace(draw: st.DrawFn) -> tuple[list[dict[str, object]], list[str], int]:
    """A submit, optional resting fills, and one enabled terminal action."""
    qty = draw(st.integers(2, 12))
    mode = draw(st.sampled_from(("fill", "cancel", "reject", "expire")))
    events: list[dict[str, object]] = [{"op": "submit", "order_id": "A", "qty": qty}]
    statuses = ["new"]
    if mode == "fill":
        parts = draw(st.integers(1, min(4, qty)))
        filled = 0
        for index, chunk in enumerate(_chunks(draw, qty, parts)):
            filled += chunk
            rest = filled < qty
            events.append(
                {
                    "op": "fill",
                    "order_id": "A",
                    "fill_id": f"f{index}",
                    "qty": chunk,
                    "rest": rest,
                }
            )
            statuses.append("partial" if rest else "filled")
        return events, statuses, qty
    room = qty - 1
    parts = draw(st.integers(0, min(3, room)))
    if parts:
        for index, chunk in enumerate(_chunks(draw, room, parts)):
            events.append(
                {
                    "op": "fill",
                    "order_id": "A",
                    "fill_id": f"f{index}",
                    "qty": chunk,
                    "rest": True,
                }
            )
            statuses.append("partial")
    events.append({"op": mode, "order_id": "A"})
    statuses.append(_TERMINAL_OP[mode])
    return events, statuses, qty


@st.composite
def _any_trace(draw: st.DrawFn) -> list[dict[str, object]]:
    count = draw(st.integers(0, 6))
    events: list[dict[str, object]] = []
    for index in range(count):
        op = draw(
            st.sampled_from(
                (
                    "submit",
                    "reject",
                    "cancel",
                    "expire",
                    "amend",
                    "fill",
                    "duplicate_fill",
                    "late_fill",
                    "checkpoint",
                    "crash",
                )
            )
        )
        order_id = draw(st.sampled_from(("A", "B")))
        qty = draw(st.integers(1, 8))
        if op in {"checkpoint", "crash"}:
            events.append({"op": op})
        elif op == "submit":
            events.append({"op": "submit", "order_id": order_id, "qty": qty})
        elif op == "amend":
            events.append({"op": "amend", "order_id": order_id, "qty": qty})
        elif op in {"reject", "cancel", "expire"}:
            events.append({"op": op, "order_id": order_id})
        else:
            events.append(
                {
                    "op": op,
                    "order_id": order_id,
                    "fill_id": f"f{index}",
                    "qty": qty,
                    "rest": draw(st.booleans()),
                }
            )
    return events


def _paper_fill(fill_id: str, qty: float, price: float, *, seconds: int) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id="o1",
        security_id="A",
        quantity=qty,
        price=price,
        fill_time=_T0 + timedelta(seconds=seconds),
    )


def _row(report: dict[str, object], order_id: str) -> dict[str, object]:
    orders = report["orders"]
    assert isinstance(orders, list)
    for row in orders:
        assert isinstance(row, dict)
        if row["order_id"] == order_id:
            return row
    raise AssertionError(order_id)


@given(case=_legal_trace())
@_FAST
def test_legal_trace_matches_new_partial_and_terminal(
    case: tuple[list[dict[str, object]], list[str], int],
) -> None:
    events, statuses, qty = case
    book = Book()
    seen: list[str] = []
    for event in events:
        book = step(book, event)
        order = book.orders["A"]
        seen.append(order.status)
        assert order.filled <= order.qty + _EPS
        assert order.qty == pytest.approx(qty)
        assert order.filled == pytest.approx(sum(order.fills.values()))
    assert seen == statuses
    assert seen[0] == "new"
    assert statuses[-1] in {"partial", "filled", "canceled", "rejected", "expired"}
    if statuses[-1] == "filled":
        assert book.orders["A"].filled == pytest.approx(qty)
    checked = check_trace(events)
    assert checked.ok
    assert checked.violations == []


@given(
    case=_legal_trace(),
    follow=st.sampled_from(("submit", "cancel", "reject", "expire", "fill")),
)
@_FAST
def test_terminal_transition_is_refused_and_leaves_the_fill(
    case: tuple[list[dict[str, object]], list[str], int],
    follow: str,
) -> None:
    events, statuses, qty = case
    if follow == "submit":
        extra: dict[str, object] = {"op": "submit", "order_id": "A", "qty": 1}
    elif follow == "fill":
        extra = {"op": "fill", "order_id": "A", "fill_id": "late", "qty": 1, "rest": False}
    else:
        extra = {"op": follow, "order_id": "A"}
    before = check_trace(events)
    assert before.ok
    filled = before.book.orders["A"].filled
    after = check_trace([*events, extra])
    assert after.ok is False
    order = after.book.orders["A"]
    assert order.status == statuses[-1]
    assert order.status in TERMINAL
    assert order.filled == pytest.approx(filled)
    assert order.filled <= qty + _EPS


@given(qty=st.integers(1, 20), extra=st.integers(1, 6))
@_FAST
def test_fill_past_order_qty_does_not_apply(qty: int, extra: int) -> None:
    result = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": qty},
            {"op": "fill", "order_id": "A", "fill_id": "f", "qty": qty + extra, "rest": False},
        ]
    )
    assert result.ok is False
    assert any("exceed" in item for item in result.violations)
    order = result.book.orders["A"]
    assert order.status == "new"
    assert order.filled == 0
    assert order.filled <= order.qty + _EPS


@given(events=_any_trace())
@_FAST
def test_arbitrary_trace_never_finishes_overfilled(events: list[dict[str, object]]) -> None:
    result = check_trace(events)
    for order in result.book.orders.values():
        assert order.filled <= order.qty + _EPS
        assert order.filled == pytest.approx(sum(order.fills.values()))
        if order.status == "partial":
            assert order.filled < order.qty - _EPS


@given(
    qty=st.integers(2, 16),
    price=st.integers(10, 40),
    side=st.sampled_from(("BUY", "SELL")),
    complete=st.booleans(),
    data=st.data(),
)
@_FAST
def test_ledger_cancel_after_fill_does_not_change_the_book(
    qty: int,
    price: int,
    side: str,
    complete: bool,
    data: st.DataObject,
) -> None:
    filled = qty if complete else data.draw(st.integers(1, qty - 1))
    ledger = OrderFillLedger(endpoint_kind="paper", starting_cash=_CASH)
    assert (
        ledger.submit(Order(order_id="o1", security_id="A", side=side, quantity=float(qty)))
        == "accepted"
    )
    assert (
        ledger.record_fill(_paper_fill("f1", float(filled), float(price), seconds=1)) == "accepted"
    )
    before = ledger.reconcile()
    verdict = ledger.cancel("o1")
    after = ledger.reconcile()
    row = _row(after, "o1")
    assert row["filled_qty"] == pytest.approx(filled)
    assert float(row["filled_qty"]) <= float(row["quantity"]) + _EPS
    assert after["computed_positions"] == before["computed_positions"]
    assert after["computed_cash"] == pytest.approx(float(before["computed_cash"]))
    assert after["live_pnl_claim"] is False
    assert after["research_only"] is True
    if complete:
        assert verdict == "already_filled"
        assert row["status"] == "filled"
        assert after["counters"]["cancels"] == 0
        assert ledger.cancel("o1") == "already_filled"
    else:
        assert verdict == "cancelled"
        assert row["status"] == "cancelled"
        assert after["counters"]["cancels"] == 1
        late = ledger.record_fill(_paper_fill("f2", 1.0, float(price), seconds=2))
        assert late == "fill_after_cancel"
        stalled = ledger.reconcile()
        assert _row(stalled, "o1")["filled_qty"] == pytest.approx(filled)
        assert stalled["computed_positions"] == before["computed_positions"]


@given(qty=st.integers(1, 16), price=st.integers(10, 40), extra=st.integers(1, 5))
@_FAST
def test_ledger_overfill_is_stored_and_not_applied(qty: int, price: int, extra: int) -> None:
    ledger = OrderFillLedger(endpoint_kind="paper", starting_cash=_CASH)
    ledger.submit(Order(order_id="o1", security_id="A", side="BUY", quantity=float(qty)))
    assert (
        ledger.record_fill(_paper_fill("too-big", float(qty + extra), float(price), seconds=1))
        == "overfill"
    )
    report = ledger.reconcile()
    assert _row(report, "o1")["status"] == "open"
    assert _row(report, "o1")["filled_qty"] == 0
    assert report["computed_positions"] == {}
    assert report["ok"] is False
    assert report["live_pnl_claim"] is False
    assert ledger.record_fill(_paper_fill("ok", float(qty), float(price), seconds=2)) == "accepted"
    filled = ledger.reconcile()
    assert _row(filled, "o1")["status"] == "filled"
    assert _row(filled, "o1")["filled_qty"] == pytest.approx(qty)


@given(qty=st.integers(1, 12), price=st.integers(10, 30))
@_FAST
def test_loopback_cancel_after_fill_returns_the_fill(qty: int, price: int) -> None:
    broker = LoopbackBroker(cash=1_000_000.0)
    order = LiveOrder("c1", "AAPL", "buy", float(qty), float(price), 1)
    ack = broker.submit(order)
    assert ack.status == "filled"
    assert ack.filled_quantity == pytest.approx(qty)
    cash = broker.cash
    positions = dict(broker.positions)
    again = broker.cancel(order.client_order_id, ack.broker_order_id)
    assert again is ack
    assert again.status == "filled"
    assert broker.cash == pytest.approx(cash)
    assert broker.positions == positions
    assert len(broker.fills(broker_to_client={}).fills) == 1


@given(qty=st.integers(1, 8), price=st.integers(10, 30))
@_FAST
def test_desk_ack_then_fill_and_cancel_after_fill_is_a_noop(qty: int, price: int) -> None:
    with TemporaryDirectory() as raw:
        desk = _open_desk(Path(raw))
        try:
            desk.arm("loopback", "", TS_NS)
            report = desk.submit(
                "aapl", "BUY", float(qty), float(price), TS_NS + 1, client_order_id="life-1"
            )
            order = report["order"]
            assert isinstance(order, dict)
            assert order["acked"] is True
            assert order["lifecycle"] == "filled"
            assert order["status"] == "filled"
            assert order["filled_qty"] == pytest.approx(qty)
            assert float(order["filled_qty"]) <= float(order["qty"]) + _EPS
            assert report["data_label"] == "SYNTHETIC"
            assert report["live_pnl_claim"] is False
            positions = dict(report["positions"])
            cash = report["cash"]
            cancelled = desk.cancel(str(order["order_id"]), TS_NS + 2)
            assert cancelled["accepted"] is False
            assert cancelled["status"] == "unknown"
            assert cancelled["positions"] == positions
            assert cancelled["cash"] == pytest.approx(float(cash))
            inbound = desk.root / "inbound"
            inbound.mkdir()
            (inbound / "late.json").write_text(
                json.dumps(
                    {
                        "schema": "live_broker_event.v1",
                        "kind": "fill",
                        "order_id": order["order_id"],
                        "event_id": "late-1",
                        "qty": 1,
                        "price": price,
                    }
                ),
                encoding="utf-8",
            )
            polled = desk.poll(TS_NS + 3)
            assert polled["kill_latched"] is True
            assert polled["positions"] == positions
            assert "fill on filled" in str(polled["halt_reason"])
            assert polled["live_pnl_claim"] is False
            blocked = desk.submit(
                "aapl", "BUY", 1.0, float(price), TS_NS + 4, client_order_id="life-2"
            )
            assert blocked["accepted"] is False
            assert blocked["kill_latched"] is True
            assert blocked["transmission"] is False
            assert blocked["positions"] == positions
        finally:
            desk.close()
