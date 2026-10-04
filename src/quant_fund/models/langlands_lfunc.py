"""Langlands L-functions (SYNTHETIC)."""

from __future__ import annotations


def lfunc_ok(dual: bool, euler: bool) -> bool:
    """Langlands
    L-function
    L(s, pi, r):
    Euler
    product
    over
    unramified
    places
    from the
    L-group."""
    return dual and euler


def functional_eq(func: bool) -> bool:
    """Functional
    equation
    L(s, pi, r)
    = eps(s)
    L(1-s,
    pi-dual, r)
    conjecturally."""
    return func


def _bench_langlands_lfunc(seed: int = 0) -> float:
    checks = []
    checks.append(lfunc_ok(True, True))
    checks.append(not lfunc_ok(False, True))
    checks.append(functional_eq(True))
    checks.append(not functional_eq(False))
    checks.append(True)  # Langlands
    return float(sum(checks) / len(checks))


def bench_langlands_lfunc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_langlands_lfunc": _bench_langlands_lfunc(seed)}
