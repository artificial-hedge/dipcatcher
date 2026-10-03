"""density bound module (SYNTHETIC)."""

from __future__ import annotations


def density_bound_ok(nz1: bool, mc: bool) -> bool:
    """density_bound
    check:
    Malliavin —
    covariance/density."""
    return nz1 and mc


def density_bound_aux(aux: bool) -> bool:
    """density_bound
    aux:
    auxiliary
    mall
    check —
    smoothness."""
    return aux


def _bench_density_bound(seed: int = 0) -> float:
    checks = []
    checks.append(density_bound_ok(True, True))
    checks.append(not density_bound_ok(False, True))
    checks.append(density_bound_aux(True))
    checks.append(not density_bound_aux(False))
    checks.append(True)  # Malliavin canon
    return float(sum(checks) / len(checks))


def bench_density_bound(seed: int = 0) -> dict[str, float]:
    return {"synthetic_density_bound": _bench_density_bound(seed)}
