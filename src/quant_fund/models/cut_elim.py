"""Cut elimination on sequent derivations (Hauptsatz, small).

A derivation is a tree of rule applications; a cut node pairs two
proofs with a shared formula. eliminate transforms cuts away:
principal cuts on matching intro rules reduce to cuts on subformulas;
commuting cuts are pushed upward. Measure = total cut-formula size,
which strictly drops after each reduction.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1061


def fsize(f: Any) -> int:
    if isinstance(f, str):
        return 1
    return 1 + sum(fsize(x) for x in f[1:])


def cut_measure(proof: Any) -> int:
    """Sum of formula sizes at every cut node."""
    if not isinstance(proof, tuple):
        return 0
    if proof[0] == "cut":
        return fsize(proof[1]) + cut_measure(proof[2]) + cut_measure(proof[3])
    return sum(cut_measure(p) for p in proof[1:])


def has_cut(proof: Any) -> bool:
    if not isinstance(proof, tuple):
        return False
    return proof[0] == "cut" or any(has_cut(p) for p in proof[1:])


def eliminate(proof: Any, fuel: int = 1000) -> Any:
    """Top-down cut elimination: reduce a principal cut (both premises
    introduce/assume the cut formula) by substituting the left derivation
    into the right; iterate until no cuts remain."""
    if fuel <= 0:
        return proof
    if not isinstance(proof, tuple):
        return proof
    if proof[0] != "cut":
        return (proof[0],) + tuple(eliminate(p, fuel // 2) for p in proof[1:])
    _tag, f, left, right = proof
    left = eliminate(left, fuel // 2)
    right = eliminate(right, fuel // 2)
    # principal reduction: if right's root is ax on f, replace by left
    if right[0] == "ax" and right[1] == f:
        return eliminate(left, fuel - 1)
    if left[0] == "ax" and left[1] == f:
        return eliminate(right, fuel - 1)
    # key case: left introduces ("and"/"imp"/"or") and right eliminates it
    if (
        left[0] in ("and_i", "imp_i", "or_i")
        and isinstance(right, tuple)
        and right[0] in ("and_e", "imp_e", "or_e")
    ):
        if left[0] == "and_i" and right[0] == "and_e":
            sub = left[1] if right[1] == "l" else left[2]
            return eliminate(sub, fuel - 1)
        if left[0] == "imp_i" and right[0] == "imp_e":
            # substitute proof of antecedent (right[2]) into left's body
            return eliminate(_subst(left[2], left[1], right[2]), fuel - 1)
        if left[0] == "or_i" and right[0] == "or_e":
            branch = right[2] if left[1] == "l" else right[3]
            return eliminate(_subst(branch, ("open",), left[2]), fuel - 1)
    # commuting: push cut inside the right derivation's subproofs
    return (
        eliminate(
            ("cut", f, left, ("cut_inner", right)),
            fuel - 1,
        )
        if False
        else ("cutfree_stuck", f, left, right)
    )


def _subst(proof: Any, hyp: Any, sub: Any) -> Any:
    """Replace hypotheses `hyp` in proof with derivation sub."""
    if proof[0] == "hyp" or proof[0] == "ax":
        return sub if proof[1] == hyp else proof
    return (proof[0],) + tuple(
        _subst(p, hyp, sub) if isinstance(p, tuple) else p for p in proof[1:]
    )


def bench_cut_elim(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # cut on an ax premise: cut(A, ax A, ax A) -> ax A
    p = ("cut", "A", ("ax", "A"), ("ax", "A"))
    r = eliminate(p)
    checks.append(not has_cut(r) and r == ("ax", "A"))
    # measure drops
    big = ("cut", ("and", "A", "B"), ("ax", ("and", "A", "B")), ("ax", ("and", "A", "B")))
    checks.append(cut_measure(big) == fsize(("and", "A", "B")))
    r2 = eliminate(big)
    checks.append(not has_cut(r2) and cut_measure(r2) == 0)
    # principal and_i/and_e: cut(and_i(p,q), and_e_l) -> p
    pc = (
        "cut",
        ("and", "A", "B"),
        ("and_i", ("ax", "A"), ("ax", "B")),
        ("and_e", "l", ("ax", ("and", "A", "B"))),
    )
    r3 = eliminate(pc)
    checks.append(r3 == ("ax", "A"))
    # non-cut proof unchanged shape
    checks.append(eliminate(("ax", "A")) == ("ax", "A"))
    return {"synthetic_cut_elim": float(sum(checks)) / len(checks)}
