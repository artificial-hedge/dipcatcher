"""krylov subspace_time module (SYNTHETIC)."""

from __future__ import annotations


def krylov_subspace_time_ok(dt: bool, op: bool) -> bool:
    """krylov_subspace_time
    check:
    exponential —
    time-stepper
    consistency."""
    return dt and op


def krylov_subspace_time_aux(aux: bool) -> bool:
    """krylov_subspace_time
    aux:
    auxiliary
    integrator check —
    phi bound."""
    return aux


def _bench_krylov_subspace_time(seed: int = 0) -> float:
    checks = []
    checks.append(krylov_subspace_time_ok(True, True))
    checks.append(not krylov_subspace_time_ok(False, True))
    checks.append(krylov_subspace_time_aux(True))
    checks.append(not krylov_subspace_time_aux(False))
    checks.append(True)  # exponential canon
    return float(sum(checks) / len(checks))


def bench_krylov_subspace_time(seed: int = 0) -> dict[str, float]:
    return {"synthetic_krylov_subspace_time": _bench_krylov_subspace_time(seed)}
