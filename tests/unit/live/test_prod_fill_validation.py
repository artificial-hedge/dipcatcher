from __future__ import annotations

import pytest

from quant_fund.live.ledger import LedgerFill, LiveLedger


def test_live_ledger_rejects_invalid_direct_broker_fills() -> None:
    ledger = LiveLedger(baseline_cash=1_000.0)
    ledger.note_order("client-1", expects_fill=True)
    invalid_fills = (
        LedgerFill("fill-1", "client-1", "SYNTH", "buy", -1.0, 10.0, 1),
        LedgerFill("fill-2", "client-1", "SYNTH", "buy", 1.0, float("nan"), 1),
        LedgerFill("fill-3", "client-1", "SYNTH", "buy", 1.0, float("inf"), 1),
        LedgerFill("fill-4", "client-1", "SYNTH", "buy", 1.0, 10.0, -1),
        LedgerFill("", "client-1", "SYNTH", "buy", 1.0, 10.0, 1),
    )

    for fill in invalid_fills:
        with pytest.raises(ValueError):
            ledger.record_fill(fill)

    assert ledger.fills() == []
    assert ledger.computed_cash == 1_000.0
    assert ledger.computed_positions == {}
