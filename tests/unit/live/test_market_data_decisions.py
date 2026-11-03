"""Pure quote, map, and corporate-action decisions. SYNTHETIC, not a receipt."""

from __future__ import annotations

import math

import pytest

from quant_fund.live.market_data import (
    MarketDataError,
    ReferenceSnapshot,
    assess_quote,
    bind_symbol,
    equity,
    jump_problem,
    mapping_is_bijective,
    plan_action,
    price_problem,
    quote_problems,
    require_internal,
    sequence_problem,
    size_problems,
    snapshot_problem,
)


def test_bad_prices_and_jump_band() -> None:
    assert price_problem(1.0) == ""
    for price in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        assert price_problem(price) == "bad_price"
    assert jump_problem(None, 10.0) == ""
    assert jump_problem(100.0, 125.0) == ""
    assert jump_problem(100.0, 75.0) == ""
    assert jump_problem(100.0, 125.0 + 1e-6) == "price_jump"
    assert jump_problem(100.0, 0.0) == "bad_price"


def test_crossed_locked_and_wide_quotes_are_distinct() -> None:
    assert quote_problems(101.0, 100.0, previous=100.0) == ("crossed",)
    assert quote_problems(100.0, 100.0, previous=100.0) == ("locked",)
    assert quote_problems(100.0, 110.0, previous=105.0) == ("wide_spread",)
    assert quote_problems(100.0, 105.0, previous=102.5) == ()
    assert quote_problems(0.0, 100.0, previous=100.0) == ("bad_bid",)
    assert quote_problems(100.0, float("nan"), previous=100.0) == ("bad_ask",)
    assert "price_jump" in quote_problems(130.0, 131.0, previous=100.0)


def test_sequence_faults_do_not_invent_the_next_print() -> None:
    assert sequence_problem(None, 7, snapshot=False) == ""
    assert sequence_problem(7, 8, snapshot=False) == ""
    assert sequence_problem(7, 7, snapshot=False) == "duplicate_sequence"
    assert sequence_problem(7, 4, snapshot=False) == "out_of_order"
    assert sequence_problem(7, 9, snapshot=False) == "sequence_gap"
    assert sequence_problem(7, 9, snapshot=True) == ""
    assert sequence_problem(7, -1, snapshot=True) == "bad_sequence"
    gap = assess_quote(
        bid=100.0,
        ask=101.0,
        seq=9,
        previous_seq=7,
        previous_mark=100.0,
        was_stale=False,
        snapshot=False,
    )
    assert gap.accept is False
    assert gap.stale is True
    assert gap.mark == 100.0
    assert gap.seq == 7
    nxt = assess_quote(
        bid=100.0,
        ask=101.0,
        seq=8,
        previous_seq=7,
        previous_mark=100.0,
        was_stale=True,
        snapshot=False,
    )
    assert nxt.accept is True
    assert nxt.stale is False
    assert nxt.mark == pytest.approx(100.5)
    assert nxt.seq == 8


def test_reference_snapshot_fails_closed_when_stale_or_unknown() -> None:
    snap = ReferenceSnapshot(
        version=2,
        as_of_ns=1_000,
        max_age_ns=50,
        lot_size={"AAPL": 1.0},
        tick_size={"AAPL": 0.01},
    )
    assert snapshot_problem(None, "AAPL", 1_000) == ""
    assert snapshot_problem(snap, "AAPL", 1_000) == ""
    assert snapshot_problem(snap, "AAPL", 1_050) == ""
    assert snapshot_problem(snap, "AAPL", 1_051) == "stale_reference"
    assert snapshot_problem(snap, "AAPL", 999) == "stale_reference"
    assert snapshot_problem(snap, "MSFT", 1_000) == "unknown_reference"
    broken = ReferenceSnapshot(0, 1_000, 50, {"AAPL": 1.0}, {"AAPL": 0.01})
    assert snapshot_problem(broken, "AAPL", 1_000) == "bad_reference_version"
    bad_lot = ReferenceSnapshot(1, 1_000, 50, {"AAPL": 0.0}, {"AAPL": 0.01})
    assert snapshot_problem(bad_lot, "AAPL", 1_000) == "bad_reference"
    assert size_problems(2.0, 100.0, 1.0, 0.01) == ()
    assert size_problems(2.5, 100.005, 1.0, 0.01) == ("lot_size", "tick_size")


def test_symbol_map_is_bijective_and_unknown_symbols_raise() -> None:
    mapping: dict[str, str] = {}
    bind_symbol(mapping, "aapl", "apl")
    bind_symbol(mapping, "AAPL", "APL")
    assert mapping == {"AAPL": "APL"}
    assert mapping_is_bijective(mapping)
    with pytest.raises(MarketDataError, match="already bound"):
        bind_symbol(mapping, "AAPL", "MSFT")
    with pytest.raises(MarketDataError, match="already bound"):
        bind_symbol(mapping, "IBM", "APL")
    assert require_internal(mapping, "apl") == "AAPL"
    with pytest.raises(MarketDataError, match="unknown broker symbol"):
        require_internal(mapping, "ZZZZ")
    assert mapping_is_bijective({})


def _book(qty: float = 100.0, mark: float = 100.0, cash: float = 1_000_000.0):
    return cash, {"AAPL": qty}, {"AAPL": mark}


def test_corporate_actions_preserve_equity_or_halt() -> None:
    cash, positions, marks = _book()
    before = equity(cash, positions, marks)
    split = plan_action(
        kind="split",
        symbol="AAPL",
        cash=cash,
        positions=positions,
        marks=marks,
        working=False,
        ratio=2.0,
        qty_cap=20_000.0,
    )
    assert split.apply is True
    assert split.halt is False
    assert split.positions["AAPL"] == pytest.approx(200.0)
    assert split.marks["AAPL"] == pytest.approx(50.0)
    assert equity(split.cash, split.positions, split.marks) == pytest.approx(before)
    reverse = plan_action(
        kind="reverse_split",
        symbol="AAPL",
        cash=cash,
        positions=positions,
        marks=marks,
        working=False,
        ratio=0.5,
        qty_cap=20_000.0,
    )
    assert reverse.apply is True
    assert reverse.positions["AAPL"] == pytest.approx(50.0)
    assert reverse.marks["AAPL"] == pytest.approx(200.0)
    assert equity(reverse.cash, reverse.positions, reverse.marks) == pytest.approx(before)
    bad_reverse = plan_action(
        kind="reverse_split",
        symbol="AAPL",
        cash=cash,
        positions=positions,
        marks=marks,
        working=False,
        ratio=2.0,
    )
    assert bad_reverse.apply is False
    assert bad_reverse.reason == "bad_ratio"
    dividend = plan_action(
        kind="dividend",
        symbol="AAPL",
        cash=cash,
        positions=positions,
        marks=marks,
        working=False,
        cash_per_share=1.0,
    )
    assert dividend.apply is True
    assert dividend.cash == pytest.approx(cash + 100.0)
    assert dividend.marks["AAPL"] == pytest.approx(99.0)
    assert dividend.cash_delta == pytest.approx(100.0)
    assert equity(dividend.cash, dividend.positions, dividend.marks) == pytest.approx(before)
    renamed = plan_action(
        kind="symbol_change",
        symbol="AAPL",
        cash=cash,
        positions=positions,
        marks=marks,
        working=False,
        new_symbol="meta",
    )
    assert renamed.apply is True
    assert renamed.new_symbol == "META"
    assert "AAPL" not in renamed.positions
    assert renamed.positions["META"] == pytest.approx(100.0)
    assert equity(renamed.cash, renamed.positions, renamed.marks) == pytest.approx(before)
    clash = plan_action(
        kind="symbol_change",
        symbol="AAPL",
        cash=cash,
        positions=positions,
        marks={**marks, "META": 10.0},
        working=False,
        new_symbol="META",
    )
    assert clash.apply is False
    assert clash.reason == "symbol_collision"
    halted = plan_action(
        kind="delisting",
        symbol="AAPL",
        cash=cash,
        positions=positions,
        marks=marks,
        working=False,
    )
    assert halted.apply is False
    assert halted.halt is True
    assert halted.reason == "delisting"
    assert halted.positions == positions
    assert halted.marks == marks
    working = plan_action(
        kind="split",
        symbol="AAPL",
        cash=cash,
        positions=positions,
        marks=marks,
        working=True,
        ratio=2.0,
    )
    assert working.apply is False
    assert working.reason == "working_orders"
    capped = plan_action(
        kind="split",
        symbol="AAPL",
        cash=cash,
        positions={"AAPL": 15_000.0},
        marks=marks,
        working=False,
        ratio=2.0,
        qty_cap=20_000.0,
    )
    assert capped.apply is False
    assert capped.reason == "position_limit"
    assert capped.positions["AAPL"] == pytest.approx(15_000.0)
    assert math.isnan(equity(cash, {"AAPL": 1.0}, {}))
