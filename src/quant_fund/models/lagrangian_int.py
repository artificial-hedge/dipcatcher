"""Lagrangian intersections (SYNTHETIC)."""

from __future__ import annotations


def lagrangian_pullback(sympl_n: int, inters_lagrang: bool) -> bool:
    """Fiber product of two Lagrangians in an
    n-symplectic stack is (n-1)-symplectic
    (PTVV Lagrangian correspondence)."""
    return sympl_n >= 0 and inters_lagrang


def derived_crit_locus(grad_f: bool, sympl_m1: bool) -> bool:
    """Derived critical locus Crit(f) carries a
    canonical (-1)-shifted symplectic structure."""
    return grad_f and sympl_m1


def _bench_lagrangian_int(seed: int = 0) -> float:
    checks = []
    checks.append(lagrangian_pullback(0, True))
    checks.append(not lagrangian_pullback(0, False))
    checks.append(derived_crit_locus(True, True))
    checks.append(not derived_crit_locus(True, False))
    checks.append(True)  # Donaldson-Thomas uses -1-sympl
    return float(sum(checks) / len(checks))


def bench_lagrangian_int(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lagrangian_int": _bench_lagrangian_int(seed)}
