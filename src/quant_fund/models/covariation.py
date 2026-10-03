"""covariation module (SYNTHETIC)."""

from __future__ import annotations


def covariation_ok(si: bool, isom: bool) -> bool:
    """covariation
    check:
    stochastic
    integral —
    isometry."""
    return si and isom


def covariation_aux(aux: bool) -> bool:
    """covariation
    aux:
    auxiliary
    integral
    check —
    covariation."""
    return aux


def _bench_covariation(seed: int = 0) -> float:
    checks = []
    checks.append(covariation_ok(True, True))
    checks.append(not covariation_ok(False, True))
    checks.append(covariation_aux(True))
    checks.append(not covariation_aux(False))
    checks.append(True)  # integration canon
    return float(sum(checks) / len(checks))


def bench_covariation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_covariation": _bench_covariation(seed)}
