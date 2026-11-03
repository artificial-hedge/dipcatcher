"""Inclusive hard-limit branches: ceilings, missing marks, and cancel vs order."""

from __future__ import annotations

from quant_fund.pretrade.codes import (
    COLLAR,
    CONCENTRATION,
    GROSS,
    KILL,
    KIND_CANCEL,
    KIND_ORDER,
    KIND_REPLACE,
    MAX_NOTIONAL,
    MAX_QTY,
    NET,
    POSITION_NOTIONAL,
    POSITION_QTY,
    STALE,
    decision_allowed,
    reason_names,
)
from quant_fund.pretrade.spec import (
    HardView,
    hard_limit_bits,
    notional_basis,
    pred_collar,
    pred_concentration,
    pred_gross,
    pred_max_notional,
    pred_max_qty,
    pred_net,
    pred_position_notional,
    pred_position_qty,
)


def _view(**overrides: float | int) -> HardView:
    base: dict[str, float | int] = {
        "qty": 1.0,
        "px": 100.0,
        "side": 1,
        "is_limit": 1,
        "pos": 0.0,
        "ref": 100.0,
        "ref_ok": 1,
        "hi": 110.0,
        "lo": 90.0,
        "gross": 0.0,
        "net": 0.0,
        "max_qty": 10.0,
        "max_notional": 1_000_000.0,
        "max_pos_qty": 10_000.0,
        "max_pos_notional": 1_000_000.0,
        "max_gross": 1_000_000.0,
        "max_net": 1_000_000.0,
        "name_limit": 1_000_000.0,
    }
    base.update(overrides)
    return HardView(**base)  # type: ignore[arg-type]


def test_quantity_and_notional_ceilings_are_inclusive() -> None:
    at_qty = _view(qty=10.0, max_qty=10.0)
    assert pred_max_qty(at_qty) == 0
    over_qty = _view(qty=10.0 + 1e-9, max_qty=10.0)
    assert pred_max_qty(over_qty) == MAX_QTY
    assert hard_limit_bits(over_qty) == MAX_QTY

    at_notional = _view(qty=2.0, px=50.0, ref=50.0, max_notional=100.0)
    assert pred_max_notional(at_notional) == 0
    over_notional = _view(qty=2.0, px=50.0, ref=50.0, max_notional=99.0)
    assert pred_max_notional(over_notional) == MAX_NOTIONAL


def test_missing_mark_does_not_invent_a_higher_notional_basis() -> None:
    assert notional_basis(10.0, 100.0, 1) == 100.0
    assert notional_basis(100.0, 10.0, 1) == 100.0
    assert notional_basis(10.0, 100.0, 0) == 10.0

    conservative = _view(qty=3.0, px=10.0, ref=100.0, ref_ok=1, max_notional=50.0)
    assert pred_max_notional(conservative) == MAX_NOTIONAL
    unmarked = _view(qty=3.0, px=10.0, ref=100.0, ref_ok=0, max_notional=50.0)
    assert pred_max_notional(unmarked) == 0
    assert hard_limit_bits(unmarked) == 0


def test_collar_skips_market_orders_and_a_missing_mark() -> None:
    outside = {"px": 200.0, "hi": 110.0, "lo": 90.0}
    assert pred_collar(_view(**outside)) == COLLAR
    assert pred_collar(_view(px=80.0, hi=110.0, lo=90.0)) == COLLAR
    assert pred_collar(_view(px=110.0)) == 0
    assert pred_collar(_view(px=90.0)) == 0
    assert pred_collar(_view(is_limit=0, **outside)) == 0
    assert pred_collar(_view(ref_ok=0, **outside)) == 0


def test_position_quantity_and_notional_both_signs() -> None:
    assert pred_position_qty(_view(qty=10.0, pos=0.0, max_pos_qty=10.0)) == 0
    long_breach = _view(qty=6.0, pos=5.0, side=1, max_pos_qty=10.0)
    short_breach = _view(qty=6.0, pos=-5.0, side=-1, max_pos_qty=10.0)
    assert pred_position_qty(long_breach) == POSITION_QTY
    assert pred_position_qty(short_breach) == POSITION_QTY
    assert hard_limit_bits(long_breach) == POSITION_QTY

    assert (
        pred_position_notional(_view(ref_ok=0, qty=10.0, pos=10.0, ref=100.0, max_pos_notional=1.0))
        == 0
    )
    long_n = _view(qty=1.0, pos=0.0, side=1, ref=100.0, max_pos_notional=50.0)
    short_n = _view(qty=1.0, pos=0.0, side=-1, ref=100.0, max_pos_notional=50.0)
    at_n = _view(qty=1.0, pos=0.0, side=1, ref=50.0, max_pos_notional=50.0)
    assert pred_position_notional(long_n) == POSITION_NOTIONAL
    assert pred_position_notional(short_n) == POSITION_NOTIONAL
    assert pred_position_notional(at_n) == 0


def test_gross_net_and_concentration_fail_closed_without_a_mark() -> None:
    assert pred_gross(_view(ref_ok=0, gross=0.0, pos=10.0, ref=10.0, qty=10.0, side=-1)) == 0
    assert pred_net(_view(ref_ok=0, net=0.0, qty=5.0, ref=100.0, max_net=1.0)) == 0
    assert pred_concentration(_view(ref_ok=0, qty=5.0, ref=100.0, name_limit=1.0)) == 0

    # Book gross does not contain the current name, so flattening it would go negative.
    negative_gross = _view(gross=0.0, pos=10.0, qty=10.0, side=-1, ref=10.0, max_gross=1_000_000.0)
    assert pred_gross(negative_gross) == GROSS
    at_gross = _view(gross=0.0, pos=0.0, qty=1.0, side=1, ref=100.0, max_gross=100.0)
    over_gross = _view(gross=0.0, pos=0.0, qty=1.0, side=1, ref=100.0, max_gross=99.0)
    assert pred_gross(at_gross) == 0
    assert pred_gross(over_gross) == GROSS

    at_long = _view(net=0.0, qty=1.0, side=1, ref=40.0, max_net=40.0)
    over_long = _view(net=0.0, qty=1.0, side=1, ref=41.0, max_net=40.0)
    over_short = _view(net=0.0, qty=1.0, side=-1, ref=41.0, max_net=40.0)
    assert pred_net(at_long) == 0
    assert pred_net(over_long) == NET
    assert pred_net(over_short) == NET

    at_name = _view(qty=1.0, pos=0.0, ref=25.0, name_limit=25.0)
    over_name = _view(qty=1.0, pos=0.0, ref=25.0, name_limit=24.0)
    assert pred_concentration(at_name) == 0
    assert pred_concentration(over_name) == CONCENTRATION
    assert hard_limit_bits(over_name) == CONCENTRATION


def test_hard_limit_bits_or_together_in_any_predicate_order() -> None:
    view = _view(qty=11.0, max_qty=10.0, px=200.0, hi=110.0, lo=90.0)
    spec = hard_limit_bits(view)
    assert spec == MAX_QTY | COLLAR
    assert hard_limit_bits(view, tuple(range(8))[::-1]) == spec
    assert hard_limit_bits(view, ()) == 0


def test_cancels_ignore_size_limits_and_orders_do_not() -> None:
    assert decision_allowed(0, KIND_ORDER) is True
    assert decision_allowed(0, KIND_REPLACE) is True
    assert decision_allowed(MAX_QTY | POSITION_NOTIONAL, KIND_ORDER) is False
    assert decision_allowed(MAX_QTY | POSITION_NOTIONAL, KIND_CANCEL) is True
    assert decision_allowed(KILL, KIND_CANCEL) is False
    assert decision_allowed(STALE, KIND_REPLACE) is False
    assert reason_names(0) == ()
    assert reason_names(KILL | MAX_QTY) == ("kill_switch", "max_order_quantity")
