from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from quant_fund.live.ledger import LedgerFill, LiveLedger


def _ledger() -> LiveLedger:
    ledger = LiveLedger(baseline_cash=1_000.0)
    ledger.note_order("order-1", expects_fill=True)
    return ledger


def _fill(
    *,
    fill_id: str = "venue-fill-1",
    client_order_id: str = "order-1",
    symbol: str = "AAPL",
    side: str = "buy",
    quantity: float = 2.0,
    price: float = 11.0,
    fill_time_ns: int = 20,
) -> LedgerFill:
    return LedgerFill(
        fill_id=fill_id,
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        fill_time_ns=fill_time_ns,
    )


def _provisional(
    *,
    fill_id: str = "order:broker-1:cum",
    quantity: float = 1.0,
    price: float = 10.0,
    fill_time_ns: int = 10,
) -> LedgerFill:
    return _fill(
        fill_id=fill_id,
        quantity=quantity,
        price=price,
        fill_time_ns=fill_time_ns,
    )


def _assert_book(
    ledger: LiveLedger,
    *,
    cash: float,
    quantity: float,
    fills: list[LedgerFill],
) -> None:
    assert ledger.computed_cash == pytest.approx(cash)
    assert ledger.computed_positions == {"AAPL": pytest.approx(quantity)}
    assert ledger.fills() == fills


def test_exact_real_fill_replay_is_idempotent_and_clean() -> None:
    ledger = _ledger()
    first = _fill()

    assert ledger.record_fill(first) == "accepted"
    assert ledger.record_fill(first) == "duplicate_fill"

    _assert_book(ledger, cash=978.0, quantity=2.0, fills=[first])
    report = ledger.reconcile()
    assert report["ok"] is True
    assert report["duplicate_fills"] == 1
    assert report["conflicting_fills"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_order_id", "order-2"),
        ("symbol", "MSFT"),
        ("side", "sell"),
        ("quantity", 3.0),
        ("price", 12.0),
        ("fill_time_ns", 19),
    ],
)
def test_real_fill_id_with_different_payload_conflicts_without_accounting(
    field: str, value: Any
) -> None:
    ledger = _ledger()
    first = _fill()
    assert ledger.record_fill(first) == "accepted"
    conflicting = replace(first, **{field: value})

    assert ledger.record_fill(conflicting) == "conflicting_fill"
    assert ledger.record_fill(conflicting) == "conflicting_fill"

    _assert_book(ledger, cash=978.0, quantity=2.0, fills=[first])
    report = ledger.reconcile()
    assert report["ok"] is False
    assert report["duplicate_fills"] == 0
    assert report["conflicting_fills"] == 2
    assert report["out_of_order_fills"] == 0


def test_cumulative_updates_replace_accounting_and_historical_replays_do_not() -> None:
    ledger = _ledger()
    first = _provisional()
    second = replace(first, quantity=3.0, price=11.0, fill_time_ns=20)
    third = replace(first, quantity=4.0, price=12.5, fill_time_ns=30)

    assert ledger.record_fill(first) == "accepted"
    assert ledger.record_fill(first) == "duplicate_fill"
    assert ledger.record_fill(second) == "replaced_provisional"
    _assert_book(ledger, cash=967.0, quantity=3.0, fills=[second])
    assert ledger.record_fill(first) == "duplicate_fill"
    _assert_book(ledger, cash=967.0, quantity=3.0, fills=[second])
    assert ledger.record_fill(third) == "replaced_provisional"
    assert ledger.record_fill(second) == "duplicate_fill"
    assert ledger.record_fill(third) == "duplicate_fill"

    _assert_book(ledger, cash=950.0, quantity=4.0, fills=[third])
    report = ledger.reconcile()
    assert report["ok"] is True
    assert report["duplicate_fills"] == 4
    assert report["conflicting_fills"] == 0


def test_cumulative_correction_can_reduce_quantity_without_double_counting() -> None:
    ledger = _ledger()
    first = _provisional(quantity=5.0, price=10.0)
    correction = replace(first, quantity=2.0, price=20.0, fill_time_ns=11)

    assert ledger.record_fill(first) == "accepted"
    assert ledger.record_fill(correction) == "replaced_provisional"

    _assert_book(ledger, cash=960.0, quantity=2.0, fills=[correction])


@pytest.mark.parametrize(
    ("field", "value"),
    [("client_order_id", "order-2"), ("symbol", "MSFT"), ("side", "sell")],
)
def test_cumulative_id_cannot_change_order_identity(field: str, value: Any) -> None:
    ledger = _ledger()
    first = _provisional(quantity=2.0)
    assert ledger.record_fill(first) == "accepted"

    assert ledger.record_fill(replace(first, **{field: value})) == "conflicting_fill"

    _assert_book(ledger, cash=980.0, quantity=2.0, fills=[first])
    report = ledger.reconcile()
    assert report["ok"] is False
    assert report["conflicting_fills"] == 1
    assert report["duplicate_fills"] == 0


def test_new_cumulative_id_for_same_order_replaces_prior_provisional() -> None:
    ledger = _ledger()
    first = _provisional(quantity=2.0, price=10.0)
    second = _provisional(fill_id="order:broker-2:cum", quantity=4.0, price=12.0, fill_time_ns=20)

    assert ledger.record_fill(first) == "accepted"
    assert ledger.record_fill(second) == "replaced_provisional"
    assert ledger.record_fill(first) == "duplicate_fill"

    _assert_book(ledger, cash=952.0, quantity=4.0, fills=[second])


def test_real_fill_replaces_cumulative_provisional() -> None:
    ledger = _ledger()
    provisional = _provisional(quantity=2.0, price=10.0)
    real = _fill(quantity=4.0, price=12.0, fill_time_ns=30)
    assert ledger.record_fill(provisional) == "accepted"

    assert ledger.record_fill(real) == "replaced_provisional"

    _assert_book(ledger, cash=952.0, quantity=4.0, fills=[real])
    assert ledger.reconcile()["ok"] is True


def test_superseded_provisional_cannot_reappear_after_real_fill() -> None:
    ledger = _ledger()
    first = _provisional()
    second = replace(first, quantity=2.0, price=11.0, fill_time_ns=20)
    real = _fill(quantity=4.0, price=12.0, fill_time_ns=30)
    assert ledger.record_fill(first) == "accepted"
    assert ledger.record_fill(second) == "replaced_provisional"
    assert ledger.record_fill(real) == "replaced_provisional"

    assert ledger.record_fill(first) == "duplicate_fill"
    assert ledger.record_fill(second) == "duplicate_fill"
    later_ack = replace(first, quantity=5.0, price=13.0, fill_time_ns=40)
    assert ledger.record_fill(later_ack) == "duplicate_fill"

    _assert_book(ledger, cash=952.0, quantity=4.0, fills=[real])
    report = ledger.reconcile()
    assert report["duplicate_fills"] == 3
    assert report["conflicting_fills"] == 0


def test_unseen_cumulative_ack_after_real_fill_is_ignored() -> None:
    ledger = _ledger()
    real = _fill(quantity=4.0, price=12.0, fill_time_ns=30)
    late_ack = _provisional(quantity=4.0, price=12.0, fill_time_ns=30)
    assert ledger.record_fill(real) == "accepted"

    assert ledger.record_fill(late_ack) == "duplicate_fill"
    assert ledger.record_fill(late_ack) == "duplicate_fill"
    assert ledger.record_fill(replace(late_ack, symbol="MSFT")) == "conflicting_fill"

    _assert_book(ledger, cash=952.0, quantity=4.0, fills=[real])
    report = ledger.reconcile()
    assert report["duplicate_fills"] == 2
    assert report["conflicting_fills"] == 1


def test_unseen_older_cumulative_snapshot_does_not_roll_back_accounting() -> None:
    ledger = _ledger()
    current = _provisional(quantity=6.0, price=12.0, fill_time_ns=20)
    stale = replace(current, quantity=4.0, price=11.0, fill_time_ns=10)
    assert ledger.record_fill(current) == "accepted"

    assert ledger.record_fill(stale) == "duplicate_fill"

    _assert_book(ledger, cash=928.0, quantity=6.0, fills=[current])
    report = ledger.reconcile()
    assert report["duplicate_fills"] == 1
    assert report["conflicting_fills"] == 0


def test_same_timestamp_cumulative_change_conflicts_without_accounting() -> None:
    ledger = _ledger()
    current = _provisional(quantity=6.0, price=12.0, fill_time_ns=20)
    ambiguous = replace(current, quantity=4.0, price=11.0)
    assert ledger.record_fill(current) == "accepted"

    assert ledger.record_fill(ambiguous) == "conflicting_fill"

    _assert_book(ledger, cash=928.0, quantity=6.0, fills=[current])
    report = ledger.reconcile()
    assert report["duplicate_fills"] == 0
    assert report["conflicting_fills"] == 1


@pytest.mark.parametrize(("field", "value"), [("symbol", "MSFT"), ("side", "sell")])
def test_first_late_provisional_identity_mismatch_conflicts(field: str, value: Any) -> None:
    ledger = _ledger()
    real = _fill(quantity=4.0, price=12.0, fill_time_ns=30)
    late_ack = replace(_provisional(quantity=4.0, price=12.0, fill_time_ns=30), **{field: value})
    assert ledger.record_fill(real) == "accepted"

    assert ledger.record_fill(late_ack) == "conflicting_fill"

    _assert_book(ledger, cash=952.0, quantity=4.0, fills=[real])
    report = ledger.reconcile()
    assert report["duplicate_fills"] == 0
    assert report["conflicting_fills"] == 1


def test_orphan_cumulative_replacement_never_unwinds_unapplied_accounting() -> None:
    ledger = LiveLedger(baseline_cash=1_000.0)
    first = _fill(fill_id="order:broker-1:cum", client_order_id="unknown", quantity=2.0, price=10.0)
    update = replace(first, quantity=3.0, price=11.0, fill_time_ns=30)

    assert ledger.record_fill(first) == "orphan_fill"
    assert ledger.record_fill(update) == "orphan_fill"

    assert ledger.computed_cash == 1_000.0
    assert ledger.computed_positions == {}
    assert ledger.fills() == [update]


def test_order_prefixed_real_fill_is_not_treated_as_cumulative() -> None:
    ledger = _ledger()
    provisional = _provisional()
    real = _fill(fill_id="order:broker-1:execution", quantity=2.0, price=11.0)
    assert ledger.record_fill(provisional) == "accepted"

    assert ledger.record_fill(real) == "replaced_provisional"

    _assert_book(ledger, cash=978.0, quantity=2.0, fills=[real])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("side", "hold"),
        ("quantity", 0.0),
        ("quantity", -1.0),
        ("quantity", math.nan),
        ("quantity", math.inf),
        ("price", 0.0),
        ("price", -1.0),
        ("price", math.nan),
        ("price", math.inf),
        ("fill_time_ns", -1),
        ("fill_time_ns", "not-a-time"),
    ],
)
def test_malformed_direct_fill_fails_closed(field: str, value: Any) -> None:
    ledger = _ledger()
    malformed = replace(_fill(), **{field: value})

    with pytest.raises(ValueError):
        ledger.record_fill(malformed)

    assert ledger.computed_cash == 1_000.0
    assert ledger.computed_positions == {}
    assert ledger.fills() == []
    report = ledger.reconcile()
    assert report["duplicate_fills"] == 0
    assert report["conflicting_fills"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("quantity", math.nan), ("price", 0.0), ("fill_time_ns", -1)],
)
def test_malformed_cumulative_update_preserves_prior_accounting(field: str, value: Any) -> None:
    ledger = _ledger()
    first = _provisional(quantity=2.0)
    assert ledger.record_fill(first) == "accepted"

    with pytest.raises(ValueError):
        ledger.record_fill(replace(first, **{field: value}))

    _assert_book(ledger, cash=980.0, quantity=2.0, fills=[first])
    report = ledger.reconcile()
    assert report["duplicate_fills"] == 0
    assert report["conflicting_fills"] == 0
