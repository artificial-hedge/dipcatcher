"""Clock skew, duplicate fills, and overfill. Idempotent apply, no extra quantity."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.fault.support import (
    SEEDS,
    STARTING_CASH,
    TS,
    FaultBroker,
    Scheduler,
    arm,
    buy,
    open_fault_session,
)

from quant_fund.live.ledger import LedgerFill, LiveLedger


def _fill(fill_id: str, qty: float, stamp: int, *, side: str = "buy") -> LedgerFill:
    return LedgerFill(fill_id, "o", "AAPL", side, qty, 10.0, stamp)


def test_shuffled_fills_apply_once_and_stay_inside_the_order(tmp_path: Path) -> None:
    del tmp_path
    for seed in SEEDS:
        sched = Scheduler(seed)
        again = Scheduler(seed)
        order = sched.interleave(["a", "b", "c", "a"])
        assert order == again.interleave(["a", "b", "c", "a"])
        book = LiveLedger(baseline_cash=STARTING_CASH)
        book.note_order("o", expects_fill=True, quantity=10.0)
        specs = {
            "a": _fill("a", 3.0, 30),
            "b": _fill("b", 3.0, 10),
            "c": _fill("c", 4.0, 20),
        }
        for name in order:
            book.record_fill(specs[name])
        assert book.computed_positions["AAPL"] == pytest.approx(10.0)
        assert book.computed_cash == pytest.approx(STARTING_CASH - 100.0)
        assert book.computed_positions["AAPL"] <= 10.0
        report = book.reconcile()
        assert report["overfills"] == 0
        assert report["live_pnl_claim"] is False
        assert book.computed_positions["AAPL"] >= 0.0


def test_overfill_and_duplicate_do_not_move_cash() -> None:
    book = LiveLedger(baseline_cash=STARTING_CASH)
    book.note_order("o", expects_fill=True, quantity=10.0)
    assert book.record_fill(_fill("f1", 6.0, 20)) == "accepted"
    assert book.record_fill(_fill("f2", 6.0, 10)) == "overfill"
    assert book.record_fill(_fill("f2", 6.0, 10)) == "duplicate_fill"
    assert book.computed_positions["AAPL"] == pytest.approx(6.0)
    assert book.computed_cash == pytest.approx(STARTING_CASH - 60.0)
    report = book.reconcile()
    assert report["ok"] is False
    assert report["overfills"] == 1
    sell = LiveLedger(baseline_cash=STARTING_CASH)
    sell.note_order("o", expects_fill=True, quantity=4.0)
    assert sell.record_fill(_fill("s1", 4.0, 5, side="sell")) == "accepted"
    assert sell.record_fill(_fill("s2", 4.0, 6, side="sell")) == "overfill"
    assert sell.computed_positions["AAPL"] == pytest.approx(-4.0)
    assert sell.computed_cash == pytest.approx(STARTING_CASH + 40.0)


def test_session_reconcile_sorts_and_caps_the_page(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    accepted = buy(session, quantity=10.0, client_order_id="page-1")
    assert accepted["status"] == "accepted"
    assert broker.submit_count == 1
    names = Scheduler(3).interleave(["early", "late", "dup"])
    specs = {
        "early": ("e", 6.0, TS + 10),
        "late": ("l", 6.0, TS + 30),
        "dup": ("e", 6.0, TS + 10),
    }
    for name in names:
        fill_id, qty, stamp = specs[name]
        broker.deliver("page-1", qty, 100.0, stamp, fill_id)
    # A broker page can repeat an id even though deliver itself is idempotent.
    broker._page.append(broker._page[0])
    report = session.reconcile(at_ns=TS + 40)
    assert report["duplicate_in_batch"] is True
    assert report["overfills"] == 1
    assert session.kill_state["latched"] is True
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(6.0)
    assert broker.positions["AAPL"] == pytest.approx(6.0)
    assert session.ledger.computed_cash == pytest.approx(STARTING_CASH - 600.0)
    assert session.ledger.computed_cash == pytest.approx(broker.cash)
    again = session.reconcile(at_ns=TS + 41)
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(6.0)
    assert session.ledger.computed_cash == pytest.approx(again["computed_cash"])
    blocked = buy(session, quantity=1.0, client_order_id="page-2", at_ns=TS + 52)
    assert blocked["ok"] is False
    assert "kill_switch" in blocked["reasons"]
    assert broker.submit_count == 1
    assert report["live_pnl_claim"] is False
