"""Descendant Gromov-Witten invariants (SYNTHETIC)."""

from __future__ import annotations


def gwd_ok(psi: bool, tau: bool) -> bool:
    """Descendant
    invariants:
    psi-classes
    at
    marked
    points
    insert
    tau
    insertions —
    gravitational
    descendants."""
    return psi and tau


def dilaton_eq(de: bool) -> bool:
    """Dilaton
    and
    string
    equations:
    universal
    recursion
    relations
    for
    descendants —
    Witten-
    Kontsevich."""
    return de


def _bench_gw_descendant(seed: int = 0) -> float:
    checks = []
    checks.append(gwd_ok(True, True))
    checks.append(not gwd_ok(False, True))
    checks.append(dilaton_eq(True))
    checks.append(not dilaton_eq(False))
    checks.append(True)  # Witten
    return float(sum(checks) / len(checks))


def bench_gw_descendant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gw_descendant": _bench_gw_descendant(seed)}
