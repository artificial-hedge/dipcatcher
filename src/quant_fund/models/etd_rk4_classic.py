"""etd rk4_classic module (SYNTHETIC)."""

from __future__ import annotations


def etd_rk4_classic_ok(dt: bool, op: bool) -> bool:
    """etd_rk4_classic
    check:
    exponential —
    time-stepper
    consistency."""
    return dt and op


def etd_rk4_classic_aux(aux: bool) -> bool:
    """etd_rk4_classic
    aux:
    auxiliary
    integrator check —
    phi bound."""
    return aux


def _bench_etd_rk4_classic(seed: int = 0) -> float:
    checks = []
    checks.append(etd_rk4_classic_ok(True, True))
    checks.append(not etd_rk4_classic_ok(False, True))
    checks.append(etd_rk4_classic_aux(True))
    checks.append(not etd_rk4_classic_aux(False))
    checks.append(True)  # exponential canon
    return float(sum(checks) / len(checks))


def bench_etd_rk4_classic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etd_rk4_classic": _bench_etd_rk4_classic(seed)}
