"""Drift alarms at the tolerance boundary.

A cash or position gap equal to the tolerance is inside the ceiling. A gap
just above it alarms. A declaration that fails halfway stays undeclared.
These rows are correctness checks on simulated ledger state.
"""

from __future__ import annotations

import pytest

from quant_fund.execution.order_recon import OrderFillLedger

_TOLERANCE = 1e-6


def _ledger() -> OrderFillLedger:
    return OrderFillLedger(
        starting_cash=0.0,
        cash_tolerance=_TOLERANCE,
        position_tolerance=_TOLERANCE,
    )


def test_drift_equal_to_tolerance_does_not_alarm() -> None:
    cash = _ledger()
    cash.declare_state(cash=_TOLERANCE, positions={})
    cash_drift = cash.reconcile()["drift"]
    assert cash_drift["cash_drift"] == _TOLERANCE
    assert cash_drift["alarm"] is False

    above = _ledger()
    above.declare_state(cash=_TOLERANCE + 1e-9, positions={})
    assert above.reconcile()["drift"]["alarm"] is True

    position = _ledger()
    position.declare_state(cash=0.0, positions={"A": _TOLERANCE})
    position_drift = position.reconcile()["drift"]
    assert position_drift["position_drift"]["A"] == _TOLERANCE
    assert position_drift["alarm"] is False

    over = _ledger()
    over.declare_state(cash=0.0, positions={"A": _TOLERANCE + 1e-9})
    assert over.reconcile()["drift"]["alarm"] is True


def test_failed_declaration_stays_undeclared() -> None:
    ledger = _ledger()
    with pytest.raises(TypeError):
        ledger.declare_state(cash=1.0, positions={"A": object()})  # type: ignore[dict-item]
    drift = ledger.reconcile()["drift"]
    assert drift["declared"] is False
    assert drift["alarm"] is False
    assert drift["cash_drift"] is None
