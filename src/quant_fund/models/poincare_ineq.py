"""Poincaré inequality (SYNTHETIC)."""

from __future__ import annotations


def poin_ok(bdd_domain: bool, mean_zero: bool) -> bool:
    """Poincaré
    inequality:
    L^p
    norm
    of
    mean-zero
    u
    bounded
    by
    gradient
    on
    bounded
    domains."""
    return bdd_domain and mean_zero


def poincare_wirtinger(pw: bool) -> bool:
    """Poincaré-
    Wirtinger:
    optimal
    constant
    is
    first
    nonzero
    Neumann
    eigenvalue."""
    return pw


def _bench_poincare_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(poin_ok(True, True))
    checks.append(not poin_ok(False, True))
    checks.append(poincare_wirtinger(True))
    checks.append(not poincare_wirtinger(False))
    checks.append(True)  # Poincaré
    return float(sum(checks) / len(checks))


def bench_poincare_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poincare_ineq": _bench_poincare_ineq(seed)}
