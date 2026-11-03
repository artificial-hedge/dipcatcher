from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.live.ledger import LedgerFill, LiveLedger


def _ledger_with_fill() -> tuple[LiveLedger, LedgerFill]:
    ledger = LiveLedger(baseline_cash=1_000.0)
    ledger.note_order("client-1", expects_fill=True)
    fill = LedgerFill("fill-1", "client-1", "AAPL", "buy", 2.0, 50.0, 100)
    assert ledger.record_fill(fill) == "accepted"
    return ledger, fill


def test_identical_fill_redelivery_is_idempotent() -> None:
    ledger, fill = _ledger_with_fill()

    assert ledger.record_fill(fill) == "duplicate_fill"

    assert ledger.computed_cash == 900.0
    assert ledger.computed_positions == {"AAPL": 2.0}
    assert ledger.fills() == [fill]
    assert ledger.reconcile()["duplicate_fills"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_order_id", "client-2"),
        ("symbol", "MSFT"),
        ("side", "sell"),
        ("quantity", 3.0),
        ("price", 51.0),
        ("fill_time_ns", 101),
    ],
)
def test_conflicting_duplicate_fill_is_rejected_without_mutation(field: str, value: object) -> None:
    ledger, fill = _ledger_with_fill()
    before = (ledger.computed_cash, ledger.computed_positions, ledger.fills())

    with pytest.raises(ValueError, match="conflicting fill id"):
        ledger.record_fill(replace(fill, **{field: value}))

    assert (ledger.computed_cash, ledger.computed_positions, ledger.fills()) == before
    assert ledger.reconcile()["duplicate_fills"] == 0
