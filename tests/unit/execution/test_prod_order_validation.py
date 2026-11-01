from __future__ import annotations

import pytest

from quant_fund.execution.order_recon import Order, OrderFillLedger, OrderLedgerError


def test_order_intake_rejects_invalid_quantity_and_identity() -> None:
    ledger = OrderFillLedger()
    invalid_orders = (
        Order("order-1", "SYNTH", "BUY", -1.0),
        Order("order-2", "SYNTH", "BUY", float("nan")),
        Order("order-3", "SYNTH", "BUY", float("inf")),
        Order("", "SYNTH", "BUY", 1.0),
        Order("order-4", "", "BUY", 1.0),
        Order("order-5", "SYNTH", "HOLD", 1.0),  # type: ignore[arg-type]
    )

    for order in invalid_orders:
        with pytest.raises(OrderLedgerError):
            ledger.submit(order)

    assert ledger.reconcile()["n_orders"] == 0
