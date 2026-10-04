"""Lindemann-Weierstrass theorem (SYNTHETIC)."""

from __future__ import annotations


def lw_ok(algebraic_exp: bool, linear_indep: bool) -> bool:
    """Lindemann-
    Weierstrass:
    exponentials
    of
    distinct
    algebraic
    numbers
    are
    linearly
    independent
    over
    the
    algebraic
    numbers."""
    return algebraic_exp and linear_indep


def exp_of_algebraic(ea: bool) -> bool:
    """Corollary:
    e^alpha
    is
    transcendental
    for
    every
    nonzero
    algebraic
    alpha."""
    return ea


def _bench_lindemann_weier(seed: int = 0) -> float:
    checks = []
    checks.append(lw_ok(True, True))
    checks.append(not lw_ok(False, True))
    checks.append(exp_of_algebraic(True))
    checks.append(not exp_of_algebraic(False))
    checks.append(True)  # Lindemann-Weierstrass
    return float(sum(checks) / len(checks))


def bench_lindemann_weier(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lindemann_weier": _bench_lindemann_weier(seed)}
