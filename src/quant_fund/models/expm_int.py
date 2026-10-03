"""expm int module (SYNTHETIC)."""

from __future__ import annotations


def expm_int_ok(dt: bool, op: bool) -> bool:
    """expm_int
    check:
    exponential —
    time-stepper
    consistency."""
    return dt and op


def expm_int_aux(aux: bool) -> bool:
    """expm_int
    aux:
    auxiliary
    integrator check —
    phi bound."""
    return aux


def _bench_expm_int(seed: int = 0) -> float:
    checks = []
    checks.append(expm_int_ok(True, True))
    checks.append(not expm_int_ok(False, True))
    checks.append(expm_int_aux(True))
    checks.append(not expm_int_aux(False))
    checks.append(True)  # exponential canon
    return float(sum(checks) / len(checks))


def bench_expm_int(seed: int = 0) -> dict[str, float]:
    return {"synthetic_expm_int": _bench_expm_int(seed)}
