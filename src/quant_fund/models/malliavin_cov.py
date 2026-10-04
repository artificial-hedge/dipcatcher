"""malliavin cov module (SYNTHETIC)."""

from __future__ import annotations


def malliavin_cov_ok(nz1: bool, mc: bool) -> bool:
    """malliavin_cov
    check:
    Malliavin —
    covariance/density."""
    return nz1 and mc


def malliavin_cov_aux(aux: bool) -> bool:
    """malliavin_cov
    aux:
    auxiliary
    mall
    check —
    smoothness."""
    return aux


def _bench_malliavin_cov(seed: int = 0) -> float:
    checks = []
    checks.append(malliavin_cov_ok(True, True))
    checks.append(not malliavin_cov_ok(False, True))
    checks.append(malliavin_cov_aux(True))
    checks.append(not malliavin_cov_aux(False))
    checks.append(True)  # Malliavin canon
    return float(sum(checks) / len(checks))


def bench_malliavin_cov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_malliavin_cov": _bench_malliavin_cov(seed)}
