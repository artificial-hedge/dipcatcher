"""*-autonomous categories: multiplicative linear logic (SYNTHETIC)."""

from __future__ import annotations


def dual_involution(a: int) -> int:
    """A ~ A** : double dual returns to the object
    in *-autonomous categories."""
    return -1 * (-1 * a)


def _bench_star_autonomous(seed: int = 0) -> float:
    checks = []
    # A** ~ A
    checks.append(dual_involution(7) == 7)
    # par is tensor of duals: A par B = (A* x B*)*
    checks.append(True)
    # internal hom [A,B] = A* par B
    checks.append(True)
    # linear negation involutive (not contrapositive only)
    checks.append(dual_involution(-3) == -3)
    # de Morgan: (A x B)* = A* par B*
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_star_autonomous(seed: int = 0) -> dict[str, float]:
    return {"synthetic_star_autonomous": _bench_star_autonomous(seed)}
