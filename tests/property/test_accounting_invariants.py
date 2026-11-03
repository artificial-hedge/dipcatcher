"""Deterministic accounting invariants for the paper book, pretrade gate, and kill switch.

Hypothesis is derandomized (fixed seed, no example database). The examples are
bounded so the module finishes well under a minute. Fills here are simulated
research records: ``live_pnl_claim`` stays false.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.execution.costs import total_cost
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
from quant_fund.monitoring.kill_switch import (
    CANCEL_OPEN_ORDERS,
    ENABLED,
    FLATTEN_OPTIONAL,
    HALT_NEW_ORDERS,
)
from quant_fund.paper.ledger import PaperLedger
from quant_fund.pretrade.codes import HARD_LIMIT_MASK, KIND_ORDER, decision_allowed
from quant_fund.pretrade.engine import PretradeEngine
from quant_fund.pretrade.spec import HardView, hard_limit_bits
from quant_fund.schemas.errors import KillSwitchActive
from quant_fund.schemas.orders import Fill, Order, OrderSide, OrderStatus
from tests.property._books import research_config
from tests.unit.pretrade.support import TS_NS, arm, make_engine, order

_EPS = 1e-8
_FAST = settings(derandomize=True, max_examples=20, deadline=None)
_BLOCKING = (HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL)
_NAMES = ("A", "B")
_T0 = datetime(2024, 1, 2, tzinfo=UTC)
_FINITE = dict(allow_nan=False, allow_infinity=False)


def _broker() -> SimulatedBroker:
    counter = {"n": 0}

    def _ids(kind: str) -> str:
        counter["n"] += 1
        return f"{kind}-{counter['n']}"

    cfg = research_config()
    cfg.costs.participation_limit = 1.0
    return SimulatedBroker(
        config=cfg,
        initial_cash=100_000.0,
        id_factory=_ids,
    )


def _order(oid: str, sid: str, side: OrderSide, qty: float, *, limit: float | None = None) -> Order:
    return Order(
        order_id=oid,
        security_id=sid,
        symbol=sid,
        side=side,
        quantity=qty,
        signal_time=_T0,
        decision_time=_T0,
        order_time=_T0,
        status=OrderStatus.NEW,
        limit_price=limit,
    )


def _fill_fee(fill: Fill) -> float:
    return float(fill.fee + fill.spread_cost + fill.impact_cost + fill.turnover_cost)


class _Inventory:
    """Average-cost book, independent of the broker's own cash identity."""

    def __init__(self, cash: float) -> None:
        self.initial = float(cash)
        self.cash = float(cash)
        self.realized = 0.0
        self.positions: dict[str, tuple[float, float]] = {}
        self.fees = 0.0

    def apply(self, sid: str, signed: float, price: float, fee: float) -> None:
        if fee < -_EPS:
            raise AssertionError(f"negative fee {fee}")
        self.fees += fee
        self.realized -= fee
        self.cash -= signed * price + fee
        qty, avg = self.positions.get(sid, (0.0, 0.0))
        if abs(qty) <= _EPS or qty * signed > 0.0:
            new_qty = qty + signed
            avg = (abs(qty) * avg + abs(signed) * price) / abs(new_qty)
            self.positions[sid] = (new_qty, avg)
            return
        close = min(abs(signed), abs(qty))
        if qty > 0.0:
            self.realized += close * (price - avg)
        else:
            self.realized += close * (avg - price)
        new_qty = qty + signed
        if abs(new_qty) <= _EPS:
            self.positions[sid] = (0.0, 0.0)
        elif qty * new_qty > 0.0:
            self.positions[sid] = (new_qty, avg)
        else:
            self.positions[sid] = (new_qty, price)

    def unrealized(self, marks: dict[str, float]) -> float:
        total = 0.0
        for sid, (qty, avg) in self.positions.items():
            if abs(qty) <= _EPS:
                continue
            total += qty * (marks[sid] - avg)
        return total

    def nav(self, marks: dict[str, float]) -> float:
        market = 0.0
        for sid, (qty, _avg) in self.positions.items():
            market += qty * marks.get(sid, 0.0)
        return self.cash + market


def _assert_conserved(book: _Inventory, broker: SimulatedBroker, marks: dict[str, float]) -> None:
    nav = book.nav(marks)
    unrealized = book.unrealized(marks)
    assert book.cash == pytest.approx(broker.cash, abs=1e-6)
    assert nav == pytest.approx(book.initial + book.realized + unrealized, abs=1e-5)
    assert nav == pytest.approx(broker.nav(marks), abs=1e-5)
    for sid in _NAMES:
        assert book.positions.get(sid, (0.0, 0.0))[0] == pytest.approx(
            broker.shares.get(sid, 0.0), abs=1e-6
        )


def _hard_view(
    engine: PretradeEngine, sid: int, side: int, qty: float, px: float, is_limit: int
) -> HardView:
    book = engine.book
    sym = book.syms[sid]
    return HardView(
        qty=qty,
        px=px,
        side=side,
        is_limit=is_limit,
        pos=sym.pos,
        ref=sym.ref,
        ref_ok=sym.ref_ok,
        hi=sym.hi,
        lo=sym.lo,
        gross=book.gross,
        net=book.net,
        max_qty=book.max_qty,
        max_notional=book.max_notional,
        max_pos_qty=book.max_pos_qty,
        max_pos_notional=book.max_pos_notional,
        max_gross=book.max_gross,
        max_net=book.max_net,
        name_limit=book.name_limit,
    )


def _ceilings(engine: PretradeEngine) -> None:
    book = engine.book
    assert book.gross >= -_EPS
    assert book.gross <= book.max_gross + 1e-5
    assert abs(book.net) <= book.max_net + 1e-5
    registered = engine._n_reg
    for sym in book.syms[:registered]:
        assert abs(sym.pos) <= book.max_pos_qty + _EPS
        assert abs(sym.pos * sym.ref) <= book.max_pos_notional + 1e-4
        assert abs(sym.pos * sym.ref) <= book.name_limit + 1e-4
        assert sym.pos >= -_EPS


@st.composite
def _fills(
    draw: st.DrawFn,
) -> tuple[bool, list[tuple[str, OrderSide, float, float]], dict[str, float]]:
    allow_short = draw(st.booleans())
    steps = draw(
        st.lists(
            st.tuples(
                st.sampled_from(_NAMES),
                st.sampled_from((OrderSide.BUY, OrderSide.SELL)),
                st.floats(1.0, 20.0, **_FINITE),
                st.floats(20.0, 60.0, **_FINITE),
            ),
            min_size=1,
            max_size=5,
        )
    )
    marks = {sid: draw(st.floats(20.0, 60.0, **_FINITE)) for sid in _NAMES}
    return allow_short, list(steps), marks


@given(_fills())
@_FAST
def test_paper_ledger_conserves_cash_pnl_and_fees(
    script: tuple[bool, list[tuple[str, OrderSide, float, float]], dict[str, float]],
) -> None:
    """Cash + marks = equity, and equity = initial + realized + unrealized.

    Without short permission the generated sells never exceed the long, and the
    posted quantities stay non-negative. Cumulative fees never decrease.
    """
    allow_short, steps, marks = script
    broker = _broker()
    book = _Inventory(broker.initial_cash)
    seen_fees: list[float] = []
    with TemporaryDirectory() as root:
        ledger = PaperLedger(root, "inv")
        assert PaperLedger.SCHEMA_VERSION == 2
        assert ledger._meta["schema_version"] == PaperLedger.SCHEMA_VERSION
        ledger.set_meta(live_pnl_claim=False, research_only=True, label="PAPER_SIMULATED")
        assert ledger._meta["schema_version"] == 2
        for index, (sid, side, qty, price) in enumerate(steps):
            held = broker.shares.get(sid, 0.0)
            if side is OrderSide.SELL and not allow_short:
                if held < 1.0:
                    side = OrderSide.BUY
                else:
                    qty = min(qty, held)
            broker.mark({sid: price})
            before_cash = broker.cash
            before_shares = dict(broker.shares)
            rec = broker.submit(
                _order(f"o{index}", sid, side, qty),
                price=price,
                adv_dollars=1e12,
                sigma=0.02,
            )
            ledger.record_orders([rec], _T0)
            if rec.fill is None:
                assert rec.order.status is OrderStatus.REJECTED
                assert broker.cash == before_cash
                assert broker.shares == before_shares
                continue
            fee = _fill_fee(rec.fill)
            assert fee >= -_EPS
            seen_fees.append(fee)
            signed = rec.fill.quantity if side is OrderSide.BUY else -rec.fill.quantity
            book.apply(sid, signed, rec.fill.price, fee)
            if not allow_short:
                assert broker.shares.get(sid, 0.0) >= -_EPS
            _assert_conserved(book, broker, dict(broker.last_marks))

        posted_fees = [float(event["fees"]) for event in ledger._cash_events]
        assert posted_fees == pytest.approx(seen_fees)
        running = 0.0
        for fee in posted_fees:
            assert fee >= -_EPS
            previous = running
            running += fee
            assert running + _EPS >= previous
        assert running == pytest.approx(book.fees, abs=1e-6)
        cash_delta = sum(float(event["cash_delta"]) for event in ledger._cash_events)
        assert broker.cash == pytest.approx(book.initial + cash_delta, abs=1e-6)

        broker.mark(marks)
        ledger.record_snapshot(broker.snapshot(asof=_T0, prices=marks))
        _assert_conserved(book, broker, marks)
        equity = ledger._equity[-1]
        assert equity["cash"] == pytest.approx(book.cash, abs=1e-6)
        assert equity["nav"] == pytest.approx(book.nav(marks), abs=1e-5)
        posted = {row["security_id"]: row["shares"] for row in ledger._positions}
        for sid, qty in broker.shares.items():
            if abs(qty) <= 1e-12:
                assert sid not in posted
                continue
            assert posted[sid] == pytest.approx(qty, abs=1e-6)
            if not allow_short:
                assert qty >= -_EPS
        assert ledger._meta["live_pnl_claim"] is False
        assert ledger._meta["research_only"] is True


@given(
    q1=st.floats(0.0, 40.0, **_FINITE),
    q2=st.floats(0.0, 40.0, **_FINITE),
    price=st.floats(5.0, 80.0, **_FINITE),
    bps=st.floats(0.0, 25.0, **_FINITE),
    impact=st.floats(0.0, 0.5, **_FINITE),
)
@_FAST
def test_fill_fees_are_monotone_in_quantity(
    q1: float, q2: float, price: float, bps: float, impact: float
) -> None:
    lo, hi = (q1, q2) if q1 <= q2 else (q2, q1)
    cfg = research_config(
        commission_bps=bps,
        half_spread_bps=bps,
        impact_y=impact,
        bps_per_turnover=bps,
    )
    low = float(total_cost(lo, price, 1e9, 0.02, cfg.costs)["total"])
    high = float(total_cost(hi, price, 1e9, 0.02, cfg.costs)["total"])
    assert math.isfinite(low) and math.isfinite(high)
    assert low >= -_EPS
    assert high + 1e-8 >= low


@st.composite
def _approved_sequences(
    draw: st.DrawFn,
) -> tuple[float, float, list[tuple[int, int, float, float, int]]]:
    max_qty = draw(st.floats(2.0, 12.0, **_FINITE))
    max_pos = draw(st.floats(max_qty, max_qty + 18.0, **_FINITE))
    steps = draw(
        st.lists(
            st.tuples(
                st.sampled_from((0, 1)),
                st.sampled_from((1, -1)),
                st.floats(0.5, 24.0, **_FINITE),
                st.floats(99.0, 101.0, **_FINITE),
                st.sampled_from((0, 1)),
            ),
            min_size=1,
            max_size=4,
        )
    )
    return max_qty, max_pos, list(steps)


@given(_approved_sequences())
@_FAST
def test_approved_orders_stay_inside_hard_limits(
    script: tuple[float, float, list[tuple[int, int, float, float, int]]],
) -> None:
    """A sequence the gate approves never leaves a hard ceiling, and a later
    order that is larger than the quantity cap is not approved."""
    max_qty, max_pos, steps = script
    ref = 100.0
    span = max_pos * ref
    engine = make_engine(
        nav=100_000.0,
        cash=100_000.0,
        max_order_quantity=max_qty,
        max_order_notional=50_000.0,
        max_position_quantity=max_pos,
        max_position_notional=span,
        max_gross_notional=2.0 * span,
        max_net_notional=2.0 * span,
        max_name_concentration=1.0,
        price_collar_bps=100.0,
        cash_account=True,
        restrict_to_settled_cash=False,
        duplicate_window_ns=1,
        max_orders_per_window=40,
        max_messages_per_window=40,
        max_lots_per_symbol=32,
        pdt_max_day_trades=100,
    )
    symbols = (arm(engine, "A", ref=ref), arm(engine, "B", ref=ref))
    _ceilings(engine)
    ts = TS_NS
    for index, (slot, side, qty, px, is_limit) in enumerate(steps):
        ts = TS_NS + (index + 1) * 10_000_000
        sid = symbols[slot]
        view = _hard_view(engine, sid, side, qty, px, is_limit)
        spec = hard_limit_bits(view)
        before = engine.book.syms[sid].pos
        bits = engine.check(
            order(sid, side=side, qty=qty, px=px, ts_ns=ts, is_limit=is_limit),
            apply=True,
        )
        assert bits & HARD_LIMIT_MASK == spec
        if decision_allowed(bits, KIND_ORDER):
            assert spec == 0
            _ceilings(engine)
        else:
            assert engine.book.syms[sid].pos == before
            _ceilings(engine)
    breach_qty = engine.book.max_qty + 1.0
    held = engine.book.syms[symbols[0]].pos
    bits = engine.check(
        order(symbols[0], side=1, qty=breach_qty, px=ref, ts_ns=ts + 10_000_000, is_limit=1),
        apply=True,
    )
    assert not decision_allowed(bits, KIND_ORDER)
    assert bits & HARD_LIMIT_MASK
    assert engine.book.syms[symbols[0]].pos == held
    _ceilings(engine)


@given(
    state=st.sampled_from(_BLOCKING),
    human=st.booleans(),
    auto=st.booleans(),
    side=st.sampled_from((OrderSide.BUY, OrderSide.SELL)),
    qty=st.floats(1.0, 5.0, **_FINITE),
)
@_FAST
def test_kill_switch_blocks_every_order_until_human_reset(
    state: str,
    human: bool,
    auto: bool,
    side: OrderSide,
    qty: float,
) -> None:
    """A tripped switch rejects new orders and resting fills. ``set_state(ENABLED)``
    is the human reset; flatten permission does not clear the latch."""
    cfg = research_config()
    cfg.kill_switch.allow_auto_flatten = auto
    cfg.costs.participation_limit = 1.0
    counter = {"n": 0}

    def _ids(kind: str) -> str:
        counter["n"] += 1
        return f"{kind}-{counter['n']}"

    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0, id_factory=_ids)
    broker.mark({"A": 50.0})
    opened = broker.submit(
        _order("seed", "A", OrderSide.BUY, 10.0),
        price=50.0,
        adv_dollars=1e12,
        sigma=0.02,
    )
    assert opened.fill is not None
    resting = broker.submit(
        _order("rest", "A", OrderSide.BUY, 2.0, limit=40.0),
        price=50.0,
        adv_dollars=1e12,
        sigma=0.02,
        bar_open=50.0,
        bar_high=52.0,
        bar_low=48.0,
    )
    assert resting.fill is None
    assert resting.order.status is OrderStatus.ACKED
    assert "rest" in broker.open_orders

    broker.kill.set_state(state)
    with pytest.raises(KillSwitchActive):
        broker.kill.set_state("BROKEN")
    assert broker.kill.state == state
    cash = broker.cash
    shares = dict(broker.shares)
    blocked = broker.submit(
        _order("blocked", "A", side, qty),
        price=50.0,
        adv_dollars=1e12,
        sigma=0.02,
    )
    assert blocked.order.status is OrderStatus.REJECTED
    assert blocked.reject_reason == RejectReason.KILL_SWITCH.value
    assert blocked.fill is None
    assert broker.cash == cash
    assert broker.shares == shares
    swept = broker.process_bar(
        "A",
        bar_open=41.0,
        bar_high=44.0,
        bar_low=39.0,
        bar_time=_T0,
        adv_dollars=1e12,
        sigma=0.02,
    )
    assert swept == []
    assert broker.cash == cash
    assert "rest" in broker.open_orders
    with pytest.raises(KillSwitchActive):
        broker.kill.assert_new_orders_allowed()
    assert broker.kill.may_flatten(human) is (state == FLATTEN_OPTIONAL and human and not auto)
    assert broker.kill.may_cancel_open_orders() is (state in {CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL})
    still = broker.submit(
        _order("still", "A", OrderSide.BUY, 1.0),
        price=50.0,
        adv_dollars=1e12,
        sigma=0.02,
    )
    assert still.reject_reason == RejectReason.KILL_SWITCH.value
    assert broker.cash == cash

    broker.kill.set_state(ENABLED)
    broker.kill.assert_new_orders_allowed()
    resumed = broker.process_bar(
        "A",
        bar_open=41.0,
        bar_high=44.0,
        bar_low=39.0,
        bar_time=_T0,
        adv_dollars=1e12,
        sigma=0.02,
    )
    assert resumed and resumed[0].fill is not None
    assert broker.shares["A"] > shares["A"]
    broker.kill.set_state(state)
    again = broker.submit(
        _order("again", "A", OrderSide.BUY, 1.0),
        price=50.0,
        adv_dollars=1e12,
        sigma=0.02,
    )
    assert again.reject_reason == RejectReason.KILL_SWITCH.value
    assert again.fill is None
