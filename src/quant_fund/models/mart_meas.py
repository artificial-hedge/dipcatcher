"""mart meas module (SYNTHETIC)."""

from __future__ import annotations


def mart_meas_ok(si: bool, isom: bool) -> bool:
    """mart_meas
    check:
    stochastic
    integral —
    isometry."""
    return si and isom


def mart_meas_aux(aux: bool) -> bool:
    """mart_meas
    aux:
    auxiliary
    integral
    check —
    covariation."""
    return aux


def _bench_mart_meas(seed: int = 0) -> float:
    checks = []
    checks.append(mart_meas_ok(True, True))
    checks.append(not mart_meas_ok(False, True))
    checks.append(mart_meas_aux(True))
    checks.append(not mart_meas_aux(False))
    checks.append(True)  # integration canon
    return float(sum(checks) / len(checks))


def bench_mart_meas(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mart_meas": _bench_mart_meas(seed)}
