from __future__ import annotations

import pytest

from quant_fund.execution.order_recon import OrderFillLedger, OrderLedgerError


def test_non_finite_broker_declarations_cannot_disable_drift_alarms() -> None:
    ledger = OrderFillLedger(starting_cash=1_000.0)
    ledger.declare_state(cash=1_000.0, positions={})

    for cash, positions in (
        (float("nan"), {}),
        (float("inf"), {}),
        (1_000.0, {"SYNTH": float("nan")}),
        (1_000.0, {"SYNTH": float("inf")}),
    ):
        with pytest.raises(OrderLedgerError):
            ledger.declare_state(cash=cash, positions=positions)

    report = ledger.reconcile()
    assert report["ok"] is True
    assert report["drift"]["cash_drift"] == 0.0
