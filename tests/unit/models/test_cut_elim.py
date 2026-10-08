"""Unit tests for quant_fund.models.cut_elim."""

from __future__ import annotations

from quant_fund.models.cut_elim import cut_measure, eliminate, has_cut


def test_commuting_conversion_enters_subproof() -> None:
    """cut(f, and_i-proof, or_i(l, and_e(l, ax f))) must commute the
    cut inside or_i, hit the principal and_e/and_i reduction, and
    come out genuinely cut-free — previously the branch was dead
    behind ``if False`` and returned cutfree_stuck instead."""
    f = ("and", "A", "B")
    left = ("and_i", ("ax", "A"), ("ax", "B"))
    right = ("or_i", "l", ("and_e", "l", ("ax", f)))
    res = eliminate(("cut", f, left, right))
    assert res == ("or_i", "l", ("ax", "A"))
    assert not has_cut(res)
    assert cut_measure(res) == 0


def test_stuck_cut_is_reported_not_masked() -> None:
    """A cut that cannot be reduced (right premise is an ax on a
    different formula and no ax matches the cut formula) must remain
    visible to has_cut/cut_measure — previously the dead branch
    emitted a 'cutfree_stuck' node that both walked past silently."""
    res = eliminate(("cut", "A", ("ax", "B"), ("ax", "C")))
    assert has_cut(res)
    assert cut_measure(res) > 0


def test_ax_trivial_cut_eliminated() -> None:
    assert eliminate(("cut", "A", ("ax", "A"), ("ax", "A"))) == ("ax", "A")


def test_principal_and_reduction() -> None:
    f = ("and", "A", "B")
    p = ("cut", f, ("and_i", ("ax", "A"), ("ax", "B")), ("and_e", "l", ("ax", f)))
    assert eliminate(p) == ("ax", "A")
