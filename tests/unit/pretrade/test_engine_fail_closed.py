"""Fail-closed validation of mutable pre-trade state."""

from __future__ import annotations

import pytest

from quant_fund.pretrade.codes import CONCENTRATION, REG_SHO, STALE
from quant_fund.pretrade.engine import PretradeEngine
from tests.unit.pretrade.support import TS_NS, arm, make_engine, order


def _account_state(engine: PretradeEngine) -> tuple[object, ...]:
    return (
        engine.book.nav,
        engine.book.session_start,
        engine.book.peak,
        engine.book.daily_floor,
        engine.book.dd_floor,
        engine.book.mark_deadline,
        engine.book.mark_ts,
        engine.book.name_limit,
        engine.book.pdt_tight,
    )


def test_infinite_nav_cannot_disable_concentration_and_allow_an_order() -> None:
    engine = make_engine(max_name_concentration=0.001)
    sid = arm(engine)
    before = _account_state(engine)

    with pytest.raises(ValueError, match="nav must be finite"):
        engine.update_account(
            nav=float("inf"),
            mark_ts_ns=TS_NS + 1,
            peak_nav=engine.book.peak,
        )

    assert _account_state(engine) == before
    bits = engine.check(order(sid, qty=20, ts_ns=TS_NS + 1), apply=False)
    assert bits & CONCENTRATION
    assert engine.last_allowed is False


@pytest.mark.parametrize("field", ["nav", "session_start_nav", "peak_nav"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_account_updates_leave_valid_state_intact(field: str, value: float) -> None:
    engine = make_engine()
    before = _account_state(engine)
    fields: dict[str, object] = {"nav": 2_000_000.0, "mark_ts_ns": TS_NS + 1}
    fields[field] = value

    with pytest.raises(ValueError, match="finite"):
        engine.update_account(**fields)  # type: ignore[arg-type]

    assert _account_state(engine) == before


@pytest.mark.parametrize("mark_ts_ns", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_mark_timestamp_is_rejected_before_account_mutation(
    mark_ts_ns: float,
) -> None:
    engine = make_engine()
    before = _account_state(engine)

    with pytest.raises(ValueError, match="mark timestamp"):
        engine.update_account(  # type: ignore[arg-type]
            nav=2_000_000.0,
            mark_ts_ns=mark_ts_ns,
        )

    assert _account_state(engine) == before


def test_infinite_reference_fails_closed_without_poisoning_exposure() -> None:
    engine = make_engine()
    sid = arm(engine, pos=2)
    assert (engine.book.gross, engine.book.net) == (200.0, 200.0)

    engine.set_symbol(sid, pos=2, ref_px=float("inf"), ref_ts_ns=TS_NS)

    sym = engine.book.syms[sid]
    assert sym.ref_ok == 0
    assert sym.ref == 0.0
    assert (engine.book.gross, engine.book.net) == (0.0, 0.0)
    assert engine.book.mark_deadline == 0
    bits = engine.check(order(sid, ts_ns=TS_NS + 1), apply=False)
    assert bits & STALE
    assert engine.last_allowed is False


@pytest.mark.parametrize("bid", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_bid_is_missing_for_rule_201(bid: float) -> None:
    engine = make_engine(cash_account=False, restrict_to_settled_cash=False)
    sid = arm(engine, sho_restricted=True, locate_ok=True, bid=bid)

    assert engine.book.syms[sid].bid == 0.0
    bits = engine.check(order(sid, side=-1, qty=1), apply=False)
    assert bits & REG_SHO
    assert engine.last_allowed is False


def test_infinite_restore_lot_quantity_is_rejected_without_mutation() -> None:
    engine = make_engine(max_lots_per_symbol=2)
    sid = arm(engine)
    engine.restore_lot(sid, qty=1.0, settles_ns=TS_NS, unsettled_funded=True)
    sym = engine.book.syms[sid]
    before = (sym.nlot, sym.lq[0], sym.ls[0], sym.lu[0])

    with pytest.raises(ValueError, match="finite and positive"):
        engine.restore_lot(
            sid,
            qty=float("inf"),
            settles_ns=TS_NS,
            unsettled_funded=False,
        )

    assert (sym.nlot, sym.lq[0], sym.ls[0], sym.lu[0]) == before
