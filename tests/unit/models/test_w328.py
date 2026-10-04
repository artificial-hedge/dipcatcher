"""Wave-328 HoTT module unit tests."""

from __future__ import annotations

from quant_fund.models.funext_toy import funext
from quant_fund.models.hit_quotient import is_prop_truncated, quotient, quotient_eq, truncate
from quant_fund.models.hlevel_check import hlevel, is_contr
from quant_fund.models.kan_hcomp import hcomp
from quant_fund.models.path_types import j_elim, refl, transport
from quant_fund.models.univalence_toy import is_equiv, transport_ua, ua


def test_paths() -> None:
    assert transport(refl(5), lambda x: x, "e") == "e"
    assert j_elim(lambda p: "m", "b", refl(0)) == "b"


def test_hlevel() -> None:
    assert is_contr([7])
    assert hlevel([True, False]) == 0


def test_univalence() -> None:
    p = ua({0: "x", 1: "y"}, [0, 1], ["x", "y"])
    assert transport_ua(p, 0) == "x"
    assert is_equiv({0: "x", 1: "y"}, [0, 1], ["x", "y"])


def test_kan() -> None:
    e = {frozenset({0, 1}), frozenset({1, 2})}
    assert hcomp(e, 0, [(0, 1), (1, 2)]) == (0, 1, 2)


def test_funext() -> None:
    assert funext(lambda x: x + 1, lambda x: x + 1, list(range(5))) is not None
    assert funext(lambda x: x + 1, lambda x: x + 2, list(range(5))) is None


def test_hit() -> None:
    q = quotient([0, 1, 2, 3], [(0, 2), (1, 3)])
    assert quotient_eq(q, 0, 2) and not quotient_eq(q, 0, 1)
    assert is_prop_truncated(truncate([1, 2]))
