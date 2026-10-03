"""Wave-318 type-theory module unit tests."""

from __future__ import annotations

import pytest

from quant_fund.models.bidirectional_tc import BOOL, NAT, TypeError_, arrow, check, infer, prod
from quant_fund.models.dep_types import VEC
from quant_fund.models.dep_types import infer as dinfer
from quant_fund.models.nbe_eval import Neutral, apply_, eval_, normalize
from quant_fund.models.proof_kernel import ATOM, IMP, Sequent, assume, imp_intro, weaken
from quant_fund.models.tactic_engine import TacticError, run
from quant_fund.models.unify_meta import occurs, solve


def test_infer_arrow() -> None:
    assert infer(("lam", "x", NAT, ("var", "x")), {}) == arrow(NAT, NAT)


def test_infer_pair() -> None:
    p = ("pair", ("lit", 1), ("lit", True))
    assert infer(p, {}) == prod(NAT, BOOL)


def test_check_rejects() -> None:
    with pytest.raises(TypeError_):
        check(("lit", 5), BOOL, {})


def test_nbe_identity() -> None:
    idf = ("lam", "x", ("var", "x"))
    assert normalize(("app", idf, ("lit", 4))) == ("lit", 4)


def test_nbe_neutral_apply() -> None:
    v = eval_(("var", "f"), {"f": Neutral("f", ())})
    apply_(v, Neutral("y", ()))
    assert normalize(
        ("app", ("var", "f"), ("var", "y")), {"f": Neutral("f", ()), "y": Neutral("y", ())}
    ) == (
        "app",
        ("var", "f"),
        ("var", "y"),
    )


def test_dep_vec_len() -> None:
    assert dinfer(("vec", (("lit", 1), ("lit", 2))), {}) == VEC(("lit", 2))


def test_unify_basic() -> None:
    s = solve(("?", "m", (("var", "x"),)), ("app", "f", ("var", "x")))
    assert s["m"][0] == ["x"]
    assert occurs("m", ("?", "m", (("var", "x"),)))


def test_kernel_weaken() -> None:
    a, b = ATOM("A"), ATOM("B")
    sq = weaken(assume(a), b)
    assert sq.ctx == frozenset({a, b})
    k = imp_intro(imp_intro(sq, b), a)
    assert k.concl == IMP(a, IMP(b, a)) and not k.ctx


def test_tactic_simple() -> None:
    a = ATOM("A")
    sq = run(Sequent(frozenset(), IMP(a, a)), [("intro",), ("assumption",)])
    assert sq.concl == IMP(a, a)
    with pytest.raises(TacticError):
        run(Sequent(frozenset(), a), [("intro",)])
