"""Regressions for live-path fault gaps. SYNTHETIC fixtures, not market evidence.

Three bugs the fault scenarios exposed:
- an unknown broker fill did not halt the armed session
- LiveLedger applied fills past a known order quantity
- submit and cancel refreshed the quote before the staleness check
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.live.brokers import LoopbackBroker
from quant_fund.live.errors import LiveGateError
from quant_fund.live.ledger import LedgerFill, LiveLedger
from tests.unit.fault.support import STARTING_CASH, TS, FaultBroker, arm, buy, open_fault_session


def test_overfill_does_not_increase_the_position() -> None:
    book = LiveLedger(baseline_cash=1_000.0)
    book.note_order("o", expects_fill=True, quantity=10.0)
    assert book.record_fill(LedgerFill("f1", "o", "AAPL", "buy", 6.0, 10.0, 2)) == "accepted"
    assert book.record_fill(LedgerFill("f2", "o", "AAPL", "buy", 6.0, 10.0, 3)) == "overfill"
    assert book.computed_positions["AAPL"] == pytest.approx(6.0)
    assert book.computed_cash == pytest.approx(940.0)
    assert book.reconcile()["overfills"] == 1


def test_unknown_fill_halts_and_stale_quote_blocks_until_mark(tmp_path: Path) -> None:
    broker = FaultBroker()
    root = tmp_path / "halt"
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)
    broker.drop_ack = True
    broker.drop_recover = True
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=1.0, client_order_id="known")
    broker.deliver("known", 1.0, 100.0, TS, "known-fill")
    broker.drop_ack = False
    broker.drop_recover = False
    resumed = open_fault_session(root, broker, venue="alpaca-paper")
    broker.inject_unknown(fill_id="ghost", quantity=3.0, price=100.0, ts=TS + 1)
    resumed.reconcile(at_ns=TS + 2)
    assert resumed.kill_state["latched"] is True
    assert resumed.ledger.computed_positions["AAPL"] == pytest.approx(1.0)
    assert broker.submit_count == 1

    loop = LoopbackBroker(cash=STARTING_CASH)
    loop.defer_fill = True
    desk = open_fault_session(tmp_path / "stale", loop, venue="loopback")
    arm(desk)
    assert buy(desk, client_order_id="work")["status"] == "accepted"
    late = TS + int(desk.pretrade.limits.max_mark_age_ns) + 1
    blocked = buy(desk, quantity=1.0, client_order_id="late", at_ns=late)
    assert blocked["reasons"] == ["stale_data"]
    assert blocked["broker_called"] is False
    assert desk.orders[0]["status"] == "accepted"
    desk.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=late)
    assert desk.cancel(client_order_id="work", actor="ada", at_ns=late)["status"] == "canceled"
