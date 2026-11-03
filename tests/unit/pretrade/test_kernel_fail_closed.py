"""Fail-closed regressions for non-finite and overflowed pre-trade risk inputs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.pretrade.codes import (
    CONCENTRATION,
    GROSS,
    INTERNAL,
    KILL,
    KIND_CANCEL,
    MAX_NOTIONAL,
    NET,
    NON_FINITE,
    POSITION_NOTIONAL,
    POSITION_QTY,
    decision_allowed,
)
from quant_fund.pretrade.spec import (
    HardView,
    hard_limit_bits,
    pred_gross,
    pred_net,
)
from tests.unit.pretrade.support import TS_NS, arm, make_engine, order


def _view() -> HardView:
    return HardView(
        qty=2.0,
        px=100.0,
        side=1,
        is_limit=1,
        pos=3.0,
        ref=100.0,
        ref_ok=1,
        hi=101.0,
        lo=99.0,
        gross=300.0,
        net=300.0,
        max_qty=2.0,
        max_notional=200.0,
        max_pos_qty=5.0,
        max_pos_notional=500.0,
        max_gross=500.0,
        max_net=500.0,
        name_limit=500.0,
    )


@pytest.mark.parametrize("field", ["qty", "px"])
def test_infinite_order_fields_are_non_finite_without_latching(field: str) -> None:
    engine = make_engine()
    sid = arm(engine)
    request = replace(order(sid), **{field: float("inf")})

    bits = engine.check(request, apply=False)

    assert bits == NON_FINITE
    assert engine.book.killed == 0


def test_finite_order_overflow_denies_without_internal_latch() -> None:
    engine = make_engine()
    sid = arm(engine)

    bits = engine.check(order(sid, qty=1e308, px=1e308), apply=False)

    assert bits & NON_FINITE
    assert bits & MAX_NOTIONAL
    assert not bits & INTERNAL
    assert not decision_allowed(bits, 1)
    assert engine.book.killed == 0


def test_nonfinite_position_denies_projection_limits_and_blocks_cancels() -> None:
    engine = make_engine()
    sid = arm(engine)
    engine.book.syms[sid].pos = float("nan")

    bits = engine.check(order(sid, ts_ns=TS_NS + 1), apply=False)

    for bit in (POSITION_QTY, POSITION_NOTIONAL, CONCENTRATION, GROSS, NET):
        assert bits & bit
    assert bits & NON_FINITE
    assert bits & INTERNAL
    assert bits & KILL
    cancel = engine.check(order(sid, kind=KIND_CANCEL, is_limit=0, ts_ns=TS_NS + 2), apply=False)
    assert not decision_allowed(cancel, KIND_CANCEL)


@pytest.mark.parametrize(("field", "bit"), [("gross", GROSS), ("net", NET)])
def test_corrupted_aggregate_exposure_blocks_cancels(field: str, bit: int) -> None:
    engine = make_engine()
    sid = arm(engine)
    setattr(engine.book, field, float("nan"))

    bits = engine.check(order(sid, ts_ns=TS_NS + 1), apply=False)

    assert bits & bit
    assert bits & NON_FINITE
    assert bits & INTERNAL
    assert bits & KILL
    cancel = engine.check(order(sid, kind=KIND_CANCEL, is_limit=0, ts_ns=TS_NS + 2), apply=False)
    assert not decision_allowed(cancel, KIND_CANCEL)


@pytest.mark.parametrize(
    "field",
    [
        "qty",
        "px",
        "pos",
        "ref",
        "hi",
        "lo",
        "gross",
        "net",
        "max_qty",
        "max_notional",
        "max_pos_qty",
        "max_pos_notional",
        "max_gross",
        "max_net",
        "name_limit",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_specification_rejects_nonfinite_fields(field: str, value: float) -> None:
    view = replace(_view(), **{field: value})

    assert hard_limit_bits(view) != 0


def test_specification_overflowed_exposure_deltas_fail_closed() -> None:
    view = replace(_view(), pos=1e308, ref=1e308)

    assert pred_gross(view) == GROSS
    assert pred_net(view) == NET


def test_specification_keeps_inclusive_ceilings() -> None:
    assert hard_limit_bits(_view()) == 0
