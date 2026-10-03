"""Wave-329 proof-theory module unit tests."""

from __future__ import annotations

from quant_fund.models.cut_elim import eliminate, has_cut
from quant_fund.models.intuit_class import classical, intuit
from quant_fund.models.linear_logic import prove as lprove
from quant_fund.models.nd_check import proves
from quant_fund.models.resolution_fol import refute, unify
from quant_fund.models.sequent_prove import prove


def test_nd() -> None:
    assert proves(("imp_i", "A", ("hyp", "A")), ("imp", "A", "A"))


def test_sequent() -> None:
    assert prove(frozenset(), ("imp", "A", "A"))
    assert not prove(frozenset(), ("or", "A", ("not", "A")))


def test_cut() -> None:
    r = eliminate(("cut", "A", ("ax", "A"), ("ax", "A")))
    assert not has_cut(r)


def test_resolution() -> None:
    assert refute([frozenset({("P", ("a",), False)}), frozenset({("P", (("var", "x"),), True)})])
    assert unify(("f", ("var", "x"), "b"), ("f", "a", ("var", "y"))) == {"x": "a", "y": "b"}


def test_linear() -> None:
    assert lprove(("A", "B"), ("tensor", "A", "B"))
    assert not lprove(("A",), ("tensor", "A", "A"))
    assert lprove(("A", ("lolli", "A", "B")), "B")


def test_intuit_class() -> None:
    assert intuit(frozenset(), ("imp", "A", "A"))
    assert classical(frozenset(), ("or", "A", ("not", "A")))
    assert not intuit(frozenset(), ("or", "A", ("not", "A")))
