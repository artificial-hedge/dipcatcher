"""Tick, gap, and corporate-action invariants. SYNTHETIC, no sockets.

Hypothesis is derandomized. Nothing here is a research receipt.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quant_fund.live.market_data import assess_quote, equity, plan_action, quote_problems
from tests.unit.fault.support import FaultBroker, arm, buy, open_fault_session

_FAST = settings(derandomize=True, max_examples=40, deadline=None)
_LIVE = settings(derandomize=True, max_examples=12, deadline=None)
_ET_OPEN = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def _book(qty: float, mark: float, cash: float) -> tuple[float, dict[str, float], dict[str, float]]:
    return cash, {"AAPL": qty}, {"AAPL": mark}


@given(
    ratio=st.floats(min_value=1.000_001, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=1e-4, max_value=1e6, allow_nan=False, allow_infinity=False),
    mark=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
    cash=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
@_FAST
def test_split_restatement_preserves_equity_or_leaves_the_book(ratio, qty, mark, cash) -> None:
    """Any split ratio either keeps equity or halts without writing the book."""
    cash0, positions, marks = _book(qty, mark, cash)
    before = equity(cash0, positions, marks)
    plan = plan_action(
        kind="split",
        symbol="AAPL",
        cash=cash0,
        positions=positions,
        marks=marks,
        working=False,
        ratio=ratio,
    )
    if plan.apply:
        assert equity(plan.cash, plan.positions, plan.marks) == pytest.approx(before, abs=1e-4)
    else:
        assert plan.halt is True
        assert plan.cash == cash0
        assert plan.positions == positions
        assert plan.marks == marks


@given(
    ratio=st.floats(min_value=1e-6, max_value=0.999_999, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=1e-4, max_value=1e6, allow_nan=False, allow_infinity=False),
    mark=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
    cash=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
@_FAST
def test_reverse_split_restatement_preserves_equity_or_leaves_the_book(
    ratio, qty, mark, cash
) -> None:
    """Any reverse-split ratio either keeps equity or halts without writing the book."""
    cash0, positions, marks = _book(qty, mark, cash)
    before = equity(cash0, positions, marks)
    plan = plan_action(
        kind="reverse_split",
        symbol="AAPL",
        cash=cash0,
        positions=positions,
        marks=marks,
        working=False,
        ratio=ratio,
    )
    if plan.apply:
        assert equity(plan.cash, plan.positions, plan.marks) == pytest.approx(before, abs=1e-4)
    else:
        assert plan.halt is True
        assert plan.cash == cash0
        assert plan.positions == positions
        assert plan.marks == marks


@given(
    bid=st.one_of(
        st.just(0.0),
        st.just(-1.0),
        st.just(float("nan")),
        st.just(float("inf")),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    ),
    ask=st.one_of(
        st.just(0.0),
        st.just(float("nan")),
        st.just(float("-inf")),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    ),
    previous=st.one_of(st.none(), st.just(100.0)),
)
@_FAST
def test_invalid_tick_never_replaces_the_mark(
    bid: float, ask: float, previous: float | None
) -> None:
    """A quote the validator rejects does not become the stored mark."""
    assume(bool(quote_problems(bid, ask, previous=previous)))
    decision = assess_quote(
        bid=bid,
        ask=ask,
        seq=1,
        previous_seq=None,
        previous_mark=previous,
        was_stale=False,
        snapshot=False,
    )
    assert decision.accept is False
    if previous is None or (isinstance(previous, float) and math.isnan(previous)):
        assert decision.mark is None or (decision.mark is not None and math.isnan(decision.mark))
    else:
        assert decision.mark == previous


@given(st.sampled_from(["gap", "duplicate", "reverse"]), st.integers(min_value=2, max_value=40))
@_LIVE
def test_gap_latch_never_authorizes_an_order(kind: str, jump: int) -> None:
    """A sequence fault leaves the mark and the broker is not called."""
    with TemporaryDirectory() as raw:
        broker = FaultBroker()
        root = Path(raw)
        session = open_fault_session(root, broker, now=_ET_OPEN)
        arm(session)
        session.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=_ET_OPEN)
        first = session.ingest_quote(
            symbol="AAPL", bid=100.0, ask=100.1, seq=1, actor="ada", at_ns=_ET_OPEN + 1
        )
        assert first["ok"] is True
        anchor = float(session.marks["AAPL"])
        seq = {"gap": 1 + jump, "duplicate": 1, "reverse": 0}[kind]
        stamp = _ET_OPEN + 2_100_000_000
        quote = session.ingest_quote(
            symbol="AAPL", bid=100.0, ask=100.1, seq=seq, actor="ada", at_ns=stamp
        )
        assert quote["ok"] is False
        assert quote["stale"] is True
        assert session.marks["AAPL"] == anchor
        denied = buy(session, client_order_id=f"{kind}-{jump}", at_ns=stamp + 1, mark=anchor)
        assert denied["broker_called"] is False
        assert "stale_data" in denied["reasons"]
        assert session.marks["AAPL"] == anchor
        assert broker.submit_count == 0
