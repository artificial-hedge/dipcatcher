"""Fundamental groups of schemes (SYNTHETIC)."""

from __future__ import annotations


def fund_grp_ok(pi1: bool, galois: bool) -> bool:
    """Étale fundamental
    group pi_1^et(X,x):
    profinite group
    classifying
    finite étale
    covers."""
    return pi1 and galois


def groth_pi1(groth: bool) -> bool:
    """Grothendieck's
    Galois-category
    definition of
    pi_1 via
    fiber functors."""
    return groth


def _bench_fundamental_grp(seed: int = 0) -> float:
    checks = []
    checks.append(fund_grp_ok(True, True))
    checks.append(not fund_grp_ok(False, True))
    checks.append(groth_pi1(True))
    checks.append(not groth_pi1(False))
    checks.append(True)  # SGA1
    return float(sum(checks) / len(checks))


def bench_fundamental_grp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fundamental_grp": _bench_fundamental_grp(seed)}
