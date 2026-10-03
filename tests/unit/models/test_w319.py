"""Wave-319 proof-automation module unit tests."""

from __future__ import annotations

import pytest

from quant_fund.models.congruence_closure import Congruence
from quant_fund.models.nelson_oppen import CombinedSolver, app, c, v
from quant_fund.models.omega_lia import lia_feasible
from quant_fund.models.ring_normalize import normalize, ring_eq
from quant_fund.models.term_rewrite import confluent_on, instantiate, match
from quant_fund.models.term_rewrite import normalize as rewnorm
from quant_fund.models.tseitin_cnf import tseitin


def test_cc_congruence() -> None:
    cc = Congruence()
    cc.merge("x", "y")
    assert cc.equal(("f", "x"), ("f", "y"))
    assert not cc.equal(("f", "x"), ("f", "z"))


def test_ring_identities() -> None:
    x, y = ("var", "x"), ("var", "y")
    assert ring_eq(("mul", x, ("add", y, y)), ("mul", ("lit", 2), ("mul", x, y)))
    assert normalize(("add", x, x)) == {("x",): 2}


def test_omega_parity() -> None:
    assert not lia_feasible([("==", (2, 2), 5)], 2)
    assert lia_feasible([("==", (2, 2), 6)], 2)


def test_nelson_propagate() -> None:
    s = CombinedSolver()
    s.assume_eq(app("f", v("x")), v("y"))
    s.assume_eq(v("x"), c(1))
    s.assume_eq(app("f", c(1)), c(2))
    assert s.equal(v("y"), c(2))


def test_rewrite_inv() -> None:
    x = ("?", "x")
    rules = [(("inv", ("inv", x)), x)]
    assert rewnorm(("inv", ("inv", ("v", "a"))), rules) == ("v", "a")
    assert match(x, ("v", "a")) == {"x": ("v", "a")}
    assert instantiate(x, {"x": ("v", "a")}) == ("v", "a")


def test_tseitin_equisat() -> None:
    clauses, top, n = tseitin(("and", "x1", "x2"))
    assert (top,) in clauses
    assert n == 3  # x1, x2 + one gate var


def test_confluence_detect() -> None:
    a, b = ("v", "a"), ("v", "b")
    assert not confluent_on([(a, b), (a, ("v", "c"))], [a])
    with pytest.raises(ValueError):
        rewnorm(("v", "a"), [(("v", "a"), ("v", "a"))], limit=20)
