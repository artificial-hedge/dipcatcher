"""Wave-326 verification-3 module unit tests."""

from __future__ import annotations

from quant_fund.models.bisim_refine import bisimilar, minimize
from quant_fund.models.ctl_mc import holds
from quant_fund.models.nba_emptiness import is_empty
from quant_fund.models.parity_game import solve
from quant_fund.models.timed_automata import guard, reset, up
from quant_fund.models.wsts_cover import coverable


def test_timed() -> None:
    z = [(0.0, float("inf"))]
    z = guard(z, 0, "<=", 2.0)
    assert z is not None
    z = up(reset(z, 0))
    assert z[0] == (0.0, float("inf"))


def test_parity() -> None:
    w0, w1 = solve({0, 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: [1], 1: [0]})
    assert w1 == {0, 1}


def test_nba() -> None:
    assert not is_empty(2, {0}, {1}, {0: [1], 1: [1]})
    assert is_empty(2, {0}, {1}, {0: [1], 1: []})


def test_ctl() -> None:
    k = ({0, 1, 2}, {0: [1], 1: [2], 2: []}, {0: set(), 1: {"p"}, 2: {"p", "q"}})
    assert holds(("ef", ("atom", "q")), k, 0)
    assert not holds(("ag", ("atom", "p")), k, 0)


def test_bisim() -> None:
    assert bisimilar({0, 1, 2}, {0: "a", 1: "a", 2: "b"}, {0: [2], 1: [2], 2: []}, 0, 1)
    assert len(minimize({0, 1, 2}, {0: "a", 1: "b", 2: "c"}, {0: [1], 1: [2], 2: []})) == 3


def test_wsts() -> None:
    t1 = ((1, 0), (0, 1))
    assert coverable([t1], (1, 0), (0, 1))
    assert not coverable([t1], (1, 0), (0, 2))
