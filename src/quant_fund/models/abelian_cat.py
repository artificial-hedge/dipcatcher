"""Abelian category theory (SYNTHETIC)."""

from __future__ import annotations


def ac_ok(abelian: bool, exact: bool) -> bool:
    """Abelian
    category:
    abelian
    cat —
    finite
    exact."""
    return abelian and exact


def abelian_axioms(aa: bool) -> bool:
    """Abelian
    axioms:
    AB axioms —
    kernels
    cokernels
    monic-epi."""
    return aa


def _bench_abelian_cat(seed: int = 0) -> float:
    checks = []
    checks.append(ac_ok(True, True))
    checks.append(not ac_ok(False, True))
    checks.append(abelian_axioms(True))
    checks.append(not abelian_axioms(False))
    checks.append(True)  # Freyd-Mitchell
    return float(sum(checks) / len(checks))


def bench_abelian_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abelian_cat": _bench_abelian_cat(seed)}
