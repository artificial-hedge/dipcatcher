from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from quant_fund.execution.order_recon import Fill, Order, OrderFillLedger, OrderLedgerError

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


def _order(**overrides: Any) -> Order:
    payload: dict[str, Any] = {
        "order_id": "order-1",
        "security_id": "SEC_A",
        "side": "BUY",
        "quantity": 10.0,
        "submitted_time": T0,
        "expects_fill": False,
    }
    payload.update(overrides)
    return Order(**payload)


def _fill(**overrides: Any) -> Fill:
    payload: dict[str, Any] = {
        "fill_id": "fill-1",
        "order_id": "order-1",
        "security_id": "SEC_A",
        "quantity": 4.0,
        "price": 10.0,
        "fill_time": T0,
        "seq": 1,
    }
    payload.update(overrides)
    return Fill(**payload)


def test_identical_order_replay_remains_idempotent() -> None:
    ledger = OrderFillLedger(accept_orders=True)
    order = _order()

    assert ledger.submit(order) == "accepted"
    assert ledger.submit(order) == "duplicate_order"

    report = ledger.reconcile()
    assert report["ok"] is True
    assert report["n_orders"] == 1
    assert report["counters"]["duplicate_orders"] == 1
    assert report["counters"]["conflicting_orders"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("security_id", "SEC_B"),
        ("side", "SELL"),
        ("quantity", 99.0),
        ("submitted_time", T0 + timedelta(seconds=1)),
        ("expects_fill", True),
    ],
)
def test_order_id_reuse_with_different_payload_conflicts(field: str, value: Any) -> None:
    ledger = OrderFillLedger(accept_orders=True)
    original = _order()
    assert ledger.submit(original) == "accepted"

    with pytest.raises(OrderLedgerError, match="conflicts with a different order"):
        ledger.submit(replace(original, **{field: value}))

    report = ledger.reconcile()
    assert report["ok"] is False
    assert report["n_orders"] == 1
    assert report["orders"][0]["security_id"] == "SEC_A"
    assert report["orders"][0]["side"] == "BUY"
    assert report["orders"][0]["quantity"] == 10.0
    assert report["counters"]["duplicate_orders"] == 0
    assert report["counters"]["conflicting_orders"] == 1


def test_identical_fill_replay_remains_idempotent() -> None:
    ledger = OrderFillLedger(starting_cash=1_000.0, accept_orders=True)
    assert ledger.submit(_order(expects_fill=True)) == "accepted"
    fill = _fill()
    assert ledger.record_fill(fill) == "accepted"

    assert ledger.record_fill(fill) == "duplicate_fill"

    report = ledger.reconcile()
    assert report["computed_positions"] == {"SEC_A": 4.0}
    assert report["computed_cash"] == pytest.approx(960.0)
    assert report["n_fills"] == 1
    assert report["counters"]["duplicate_fills"] == 1
    assert report["counters"]["conflicting_fills"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("order_id", "order-2"),
        ("security_id", "SEC_B"),
        ("quantity", 5.0),
        ("price", 11.0),
        ("fill_time", T0 + timedelta(seconds=1)),
        ("seq", 2),
    ],
)
def test_fill_id_reuse_with_different_payload_conflicts_without_accounting(
    field: str, value: Any
) -> None:
    ledger = OrderFillLedger(starting_cash=1_000.0, accept_orders=True)
    assert ledger.submit(_order(expects_fill=True)) == "accepted"
    original = _fill()
    assert ledger.record_fill(original) == "accepted"

    assert ledger.record_fill(replace(original, **{field: value})) == "conflicting_fill"

    report = ledger.reconcile()
    assert report["ok"] is False
    assert report["computed_positions"] == {"SEC_A": 4.0}
    assert report["computed_cash"] == pytest.approx(960.0)
    assert report["n_fills"] == 1
    assert report["orders"][0]["filled_qty"] == 4.0
    assert report["counters"]["duplicate_fills"] == 0
    assert report["counters"]["conflicting_fills"] == 1
