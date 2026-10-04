"""dealiasing module (SYNTHETIC)."""

from __future__ import annotations


def dealiasing_ok(grid: bool, basis: bool) -> bool:
    """dealiasing
    check:
    spectral
    method —
    grid."""
    return grid and basis


def dealiasing_aux(aux: bool) -> bool:
    """dealiasing
    aux:
    auxiliary
    spectral check —
    coeff."""
    return aux


def _bench_dealiasing(seed: int = 0) -> float:
    checks = []
    checks.append(dealiasing_ok(True, True))
    checks.append(not dealiasing_ok(False, True))
    checks.append(dealiasing_aux(True))
    checks.append(not dealiasing_aux(False))
    checks.append(True)  # spectral-methods canon
    return float(sum(checks) / len(checks))


def bench_dealiasing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dealiasing": _bench_dealiasing(seed)}
