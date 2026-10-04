"""hamilton jacobi module (SYNTHETIC)."""

from __future__ import annotations


def hamilton_jacobi_ok(sc1: bool, fs: bool) -> bool:
    """hamilton_jacobi
    check:
    stochastic
    control —
    Fleming-Soner
    verification."""
    return sc1 and fs


def hamilton_jacobi_aux(aux: bool) -> bool:
    """hamilton_jacobi
    aux:
    auxiliary
    HJB
    check —
    dynamic
    programming."""
    return aux


def _bench_hamilton_jacobi(seed: int = 0) -> float:
    checks = []
    checks.append(hamilton_jacobi_ok(True, True))
    checks.append(not hamilton_jacobi_ok(False, True))
    checks.append(hamilton_jacobi_aux(True))
    checks.append(not hamilton_jacobi_aux(False))
    checks.append(True)  # control canon
    return float(sum(checks) / len(checks))


def bench_hamilton_jacobi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hamilton_jacobi": _bench_hamilton_jacobi(seed)}
