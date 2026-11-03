"""Boundary rows for hot_check. The hard-limit property samples the interior."""

from __future__ import annotations

import pytest

from quant_fund.pretrade.codes import DAILY_LOSS, MAX_QTY, NON_FINITE
from tests.unit.pretrade.support import TS_NS, arm, make_engine, order


def test_zero_quantity_is_non_finite() -> None:
    """``qty == 0`` is rejected. A positive quantity is not rejected for that reason."""
    engine = make_engine(restrict_to_settled_cash=False, cash_account=False)
    sid = arm(engine)
    assert engine.check(order(sid, qty=0.0), apply=False) & NON_FINITE
    positive = engine.check(order(sid, qty=1.0, ts_ns=TS_NS + 1), apply=False)
    assert (positive & NON_FINITE) == 0


def test_quantity_at_the_cap_is_inside_and_past_it_is_not() -> None:
    """The quantity cap is strict: ``qty == max`` is allowed, ``qty > max`` is not."""
    engine = make_engine(
        max_order_quantity=10.0,
        max_order_notional=1.0e12,
        restrict_to_settled_cash=False,
        cash_account=False,
    )
    sid = arm(engine)
    at_cap = engine.check(order(sid, qty=10.0, px=100.0), apply=False)
    assert (at_cap & MAX_QTY) == 0
    over = engine.check(order(sid, qty=10.0 + 1e-4, px=100.0, ts_ns=TS_NS + 1), apply=False)
    assert over & MAX_QTY


def test_nav_on_the_daily_floor_trips_and_just_above_does_not() -> None:
    """Daily loss uses ``nav <= floor``. Equality is a breach."""
    floor = 900.0

    def _check(nav: float) -> int:
        engine = make_engine(
            nav=1_000.0,
            cash=1_000.0,
            max_daily_loss_fraction=0.1,
            restrict_to_settled_cash=False,
            cash_account=False,
        )
        sid = arm(engine)
        engine.update_account(
            nav=nav,
            mark_ts_ns=TS_NS,
            session_start_nav=1_000.0,
            peak_nav=1_000.0,
        )
        assert engine.book.daily_floor == pytest.approx(floor)
        return engine.check(order(sid, qty=1.0, px=100.0), apply=False)

    assert _check(floor) & DAILY_LOSS
    assert (_check(floor + 0.01) & DAILY_LOSS) == 0
