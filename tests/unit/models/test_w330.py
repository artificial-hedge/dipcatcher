"""Wave-330 SMT-theory module unit tests."""

from __future__ import annotations

from quant_fund.models.array_theory import consistent_assign, holds
from quant_fund.models.bv_ops import bv_add, bv_slt, solve_enum
from quant_fund.models.diff_logic import consistent, implies, tightest
from quant_fund.models.dpllt import solve
from quant_fund.models.lia_branch import ilp_solve
from quant_fund.models.mcsat_lite import mcsat_solve


def test_diff_logic() -> None:
    assert consistent([(0, 1, 5.0)], 2)
    assert not consistent([(0, 1, -2.0), (1, 0, 1.0)], 2)
    assert implies([(0, 1, 5.0), (1, 2, 3.0)], 3, (0, 2, 8.0))
    assert abs(tightest([(0, 1, 5.0), (1, 2, 3.0)], 3)[0][2] - 8.0) < 1e-9


def test_array() -> None:
    a = ("arr", "a")
    w = ("store", a, ("const", 1), ("const", 9))
    assert holds((("read", w, ("const", 1)), ("const", 9)))
    assert consistent_assign(
        [("a", 0, 5)], [(("read", ("arr", "a"), ("const", 0)), ("const", 5), True)]
    )


def test_bv() -> None:
    assert bv_add(255, 1, 8) == 0
    assert bv_slt(0x80, 0x7F, 8)
    assert solve_enum([(lambda xs: bv_add(xs[0], 1, 4), "=", 0)], [4]) == [15]


def test_dpllt() -> None:
    assert solve([frozenset({1}), frozenset({2})], 2, lambda a: not (a.get(1) and a.get(2))) is None
    out = solve([frozenset({1}), frozenset({-1, 2})], 2, lambda a: True)
    assert out is not None and out[1] and out[2]


def test_lia() -> None:
    xs = ilp_solve([([1.0, 1.0], 5.0), ([-1.0, 1.0], -1.0)], 2)
    assert xs is not None
    assert ilp_solve([([1.0], 1.0), ([-1.0], -2.0)], 1) is None


def test_mcsat() -> None:
    sol = mcsat_solve([[("x", 0, 3)], [("x", -9, 1)]], ["x"])
    assert sol is not None and 0 <= sol["x"] <= 1
    assert mcsat_solve([[("x", -9, 1)], [("x", 2, 9)]], ["x"]) is None
