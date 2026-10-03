"""Wave-325 ownership/substructural module unit tests."""

from __future__ import annotations

import pytest

from quant_fund.models.borrow_check import check as bcheck
from quant_fund.models.capability_perm import PermError
from quant_fund.models.capability_perm import run as prun
from quant_fund.models.escape_region import EscapeError
from quant_fund.models.escape_region import check as echeck
from quant_fund.models.lifetime_outlives import check_outlives, solve
from quant_fund.models.linear_use import LinearError
from quant_fund.models.linear_use import check as lcheck
from quant_fund.models.refinement_liquid import sub


def test_nll_last_use() -> None:
    assert bcheck([("borrow", "x", "shared", "L"), ("loan_use", "L"), ("mut", "x")]) == []
    assert bcheck([("borrow", "x", "shared", "L"), ("mut", "x"), ("loan_use", "L")]) != []


def test_outlives() -> None:
    pts = solve([("eq_at", "b", 2), ("outlives", "a", "b")], ["a", "b"])
    assert pts["a"] == {2}
    assert check_outlives([("a", "b"), ("b", "c")], ("a", "c"))


def test_linear() -> None:
    lcheck(("app", "x", "z"), {"x": "lin", "z": "un"})
    with pytest.raises(LinearError):
        lcheck(("lit", 1), {"x": "lin"})


def test_region_escape() -> None:
    echeck(("letregion", "r", ("let", "t", ("at", ("lit", 1), "r"), ("lit", 5))))
    with pytest.raises(EscapeError):
        echeck(("letregion", "r", ("at", ("lit", 1), "r")))


def test_permissions() -> None:
    st = prun([("alloc", "x"), ("split", "x", "a", "b"), ("join", "a", "b", "x"), ("write", "x")])
    assert st["x"] == "U"
    with pytest.raises(PermError):
        prun([("alloc", "x"), ("alias_ro", "x", "y"), ("write", "y")])


def test_liquid() -> None:
    d = range(-5, 11)
    assert sub(("ge", "nu", 0), ("ge", "nu", -1), d)
    assert not sub(("ge", "nu", -1), ("ge", "nu", 0), d)
