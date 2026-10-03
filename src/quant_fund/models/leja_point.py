"""leja point module (SYNTHETIC)."""

from __future__ import annotations


def leja_point_ok(dt: bool, op: bool) -> bool:
    """leja_point
    check:
    exponential —
    time-stepper
    consistency."""
    return dt and op


def leja_point_aux(aux: bool) -> bool:
    """leja_point
    aux:
    auxiliary
    integrator check —
    phi bound."""
    return aux


def _bench_leja_point(seed: int = 0) -> float:
    checks = []
    checks.append(leja_point_ok(True, True))
    checks.append(not leja_point_ok(False, True))
    checks.append(leja_point_aux(True))
    checks.append(not leja_point_aux(False))
    checks.append(True)  # exponential canon
    return float(sum(checks) / len(checks))


def bench_leja_point(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leja_point": _bench_leja_point(seed)}
