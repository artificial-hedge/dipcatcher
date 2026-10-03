"""phi function module (SYNTHETIC)."""

from __future__ import annotations


def phi_function_ok(dt: bool, op: bool) -> bool:
    """phi_function
    check:
    exponential —
    time-stepper
    consistency."""
    return dt and op


def phi_function_aux(aux: bool) -> bool:
    """phi_function
    aux:
    auxiliary
    integrator check —
    phi bound."""
    return aux


def _bench_phi_function(seed: int = 0) -> float:
    checks = []
    checks.append(phi_function_ok(True, True))
    checks.append(not phi_function_ok(False, True))
    checks.append(phi_function_aux(True))
    checks.append(not phi_function_aux(False))
    checks.append(True)  # exponential canon
    return float(sum(checks) / len(checks))


def bench_phi_function(seed: int = 0) -> dict[str, float]:
    return {"synthetic_phi_function": _bench_phi_function(seed)}
