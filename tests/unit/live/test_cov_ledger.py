"""Coverage tests for the fill ledger's fail-closed and reconciliation branches.

These exercises only ledger accounting: no venue, no gateway, no network.
"""

from __future__ import annotations

import math

import pytest

from quant_fund.live.ledger import LedgerFill, LiveLedger, ledger_fill_from_dict


def _fill(
    fill_id: str = "f1",
    *,
    order: str = "o1",
    symbol: str = "AAPL",
    side: str = "buy",
    quantity: float = 1.0,
    price: float = 10.0,
    time_ns: int = 1,
) -> LedgerFill:
    return LedgerFill(
        fill_id=fill_id,
        client_order_id=order,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        fill_time_ns=time_ns,
    )


def _ledger(**kwargs: float) -> LiveLedger:
    base = {"baseline_cash": 1_000.0}
    base.update(kwargs)
    return LiveLedger(**base)


def _raw(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "fill_id": "f1",
        "client_order_id": "o1",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 2.0,
        "price": 10.0,
        "fill_time_ns": 7,
    }
    row.update(overrides)
    return row


class TestFillFromDict:
    def test_accepts_well_formed_row(self) -> None:
        fill = ledger_fill_from_dict(_raw())
        assert fill.side == "buy"
        assert fill.quantity == 2.0
        assert fill.as_dict()["fill_time_ns"] == 7

    def test_rejects_unknown_side(self) -> None:
        with pytest.raises(ValueError, match="buy or sell"):
            ledger_fill_from_dict(_raw(side="short"))

    def test_rejects_negative_fill_time(self) -> None:
        with pytest.raises(ValueError, match="fill_time_ns"):
            ledger_fill_from_dict(_raw(fill_time_ns=-1))

    @pytest.mark.parametrize("bad", [0.0, -0.5, math.nan, math.inf])
    def test_rejects_degenerate_quantity(self, bad: float) -> None:
        with pytest.raises(ValueError, match="fill quantity"):
            ledger_fill_from_dict(_raw(quantity=bad))

    @pytest.mark.parametrize("bad", [0.0, -3.0, math.nan, math.inf])
    def test_rejects_degenerate_price(self, bad: float) -> None:
        with pytest.raises(ValueError, match="fill price"):
            ledger_fill_from_dict(_raw(price=bad))


class TestConstruction:
    @pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
    def test_rejects_bad_baseline_cash(self, bad: float) -> None:
        with pytest.raises(ValueError, match="baseline cash"):
            LiveLedger(baseline_cash=bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_nonfinite_baseline_position(self, bad: float) -> None:
        with pytest.raises(ValueError, match="baseline position"):
            LiveLedger(baseline_cash=1.0, baseline_positions={"AAPL": bad})

    def test_copies_baseline_positions(self) -> None:
        positions = {"AAPL": 2.0}
        ledger = LiveLedger(baseline_cash=100.0, baseline_positions=positions)
        positions["AAPL"] = 99.0
        assert ledger.computed_positions == {"AAPL": 2.0}
        assert ledger.computed_cash == 100.0


class TestOrderExpectations:
    def test_set_expects_fill_requires_known_order(self) -> None:
        ledger = _ledger()
        with pytest.raises(ValueError, match="not in the ledger"):
            ledger.set_expects_fill("ghost", False)

    def test_toggling_expectation_clears_missing_fill(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=True)
        assert ledger.reconcile()["missing_fills"] == ["o1"]
        ledger.set_expects_fill("o1", False)
        ledger.declare(cash=ledger.computed_cash, positions=ledger.computed_positions)
        report = ledger.reconcile()
        assert report["missing_fills"] == []
        assert report["drift"]["alarm"] is False
        assert report["ok"] is True

    def test_order_not_expecting_fill_is_never_missing(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=False)
        assert ledger.reconcile()["missing_fills"] == []


class TestRecordFill:
    def test_rejects_side_outside_contract(self) -> None:
        ledger = _ledger()
        with pytest.raises(ValueError, match="buy or sell"):
            ledger.record_fill(_fill(side="hold"))

    def test_duplicate_fill_is_counted_not_reapplied(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=True)
        assert ledger.record_fill(_fill()) == "accepted"
        assert ledger.record_fill(_fill()) == "duplicate_fill"
        ledger.declare(cash=990.0, positions={"AAPL": 1.0})
        report = ledger.reconcile()
        assert report["duplicate_fills"] == 1
        assert report["computed_cash"] == 990.0
        assert report["computed_positions"] == {"AAPL": 1.0}
        # Idempotent re-delivery alone does not make a declared book unclean.
        assert report["drift"]["alarm"] is False
        assert report["ok"] is True

    def test_orphan_fill_is_flagged_and_not_applied(self) -> None:
        ledger = _ledger()
        assert ledger.record_fill(_fill(order="unknown")) == "orphan_fill"
        report = ledger.reconcile()
        assert report["orphan_fills"] == 1
        assert report["ok"] is False
        assert report["computed_cash"] == 1_000.0
        assert report["computed_positions"] == {}
        assert len(ledger.fills()) == 1

    def test_out_of_order_fill_is_flagged(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=True)
        assert ledger.record_fill(_fill("f1", time_ns=10)) == "accepted"
        assert ledger.record_fill(_fill("f2", time_ns=5)) == "out_of_order_fill"
        report = ledger.reconcile()
        assert report["out_of_order_fills"] == 1
        assert report["ok"] is False
        # Both fills still apply; out-of-order is an ordering alarm only.
        assert report["computed_positions"] == {"AAPL": 2.0}

    def test_sell_fill_debits_position_and_credits_cash(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=True)
        assert ledger.record_fill(_fill(side="sell", quantity=3.0, price=20.0)) == "accepted"
        assert ledger.computed_positions == {"AAPL": -3.0}
        assert ledger.computed_cash == 1_060.0

    def test_venue_fill_replaces_provisional(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=True)
        ledger.record_fill(_fill("order:o1", quantity=5.0, price=10.0))
        assert ledger.computed_cash == 950.0
        verdict = ledger.record_fill(_fill("f1", quantity=3.0, price=30.0, time_ns=2))
        assert verdict == "replaced_provisional"
        # Provisional cost is unwound: only the 3 @ 30 venue fill remains.
        assert ledger.computed_cash == 910.0
        assert ledger.computed_positions == {"AAPL": 3.0}
        assert [f.fill_id for f in ledger.fills()] == ["f1"]

    def test_replacement_resets_last_fill_time_to_earlier_real_fill(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=True)
        ledger.record_fill(_fill("f0", time_ns=50))
        ledger.record_fill(_fill("order:a", time_ns=100))
        ledger.record_fill(_fill("order:b", time_ns=200))
        assert ledger.record_fill(_fill("f1", time_ns=150)) == "replaced_provisional"
        # The stale provisional timestamps are gone: a fill after f0's time is
        # still in order relative to the remaining real fills.
        assert ledger.record_fill(_fill("f2", time_ns=160)) == "accepted"
        assert ledger.reconcile()["out_of_order_fills"] == 0

    def test_provisional_itself_is_never_a_replacement(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=True)
        assert ledger.record_fill(_fill("order:a", time_ns=1)) == "accepted"
        assert ledger.record_fill(_fill("order:b", time_ns=2)) == "accepted"
        assert len(ledger.fills()) == 2


class TestDeclareAndReconcile:
    def test_undeclared_reconciliation_is_not_a_clean_book(self) -> None:
        report = _ledger().reconcile()
        assert report["drift"]["declared"] is False
        assert report["drift"]["cash_drift"] is None
        assert report["drift"]["position_drift"] == {}
        assert report["drift"]["alarm"] is True
        assert report["ok"] is False
        assert report["live_pnl_claim"] is False
        assert report["research_evidence"] is False

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_nonfinite_declared_cash(self, bad: float) -> None:
        with pytest.raises(ValueError, match="declared cash"):
            _ledger().declare(cash=bad, positions={})

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_rejects_nonfinite_declared_position(self, bad: float) -> None:
        with pytest.raises(ValueError, match="declared position"):
            _ledger().declare(cash=1.0, positions={"AAPL": bad})

    def test_matching_declaration_has_no_alarm(self) -> None:
        ledger = _ledger()
        ledger.declare(cash=1_000.0, positions={})
        report = ledger.reconcile()
        assert report["drift"] == {
            "declared": True,
            "cash_drift": 0.0,
            "position_drift": {},
            "alarm": False,
        }
        assert report["ok"] is True

    def test_cash_drift_raises_alarm(self) -> None:
        ledger = _ledger()
        ledger.declare(cash=999.0, positions={})
        report = ledger.reconcile()
        assert report["drift"]["cash_drift"] == 1.0
        assert report["drift"]["alarm"] is True
        assert report["ok"] is False

    def test_position_drift_covers_both_sides_of_the_book(self) -> None:
        ledger = _ledger()
        ledger.note_order("o1", expects_fill=True)
        ledger.record_fill(_fill(quantity=2.0, price=10.0))
        ledger.declare(cash=980.0, positions={"AAPL": 1.0, "MSFT": 4.0})
        position_drift = ledger.reconcile()["drift"]["position_drift"]
        assert position_drift == {"AAPL": 1.0, "MSFT": 4.0}

    def test_drift_within_tolerance_is_clean(self) -> None:
        ledger = LiveLedger(
            baseline_cash=1_000.0,
            baseline_positions={"AAPL": 1.0},
            cash_tolerance=0.5,
            position_tolerance=0.5,
        )
        ledger.declare(cash=999.5, positions={"AAPL": 1.5})
        report = ledger.reconcile()
        assert report["drift"]["alarm"] is False
        assert report["ok"] is True

    def test_missing_and_orphan_reasons_are_reported_together(self) -> None:
        ledger = _ledger()
        ledger.note_order("o2", expects_fill=True)
        ledger.record_fill(_fill(order="o1"))
        report = ledger.reconcile()
        assert report["missing_fills"] == ["o2"]
        assert report["orphan_fills"] == 1
        assert report["ok"] is False
