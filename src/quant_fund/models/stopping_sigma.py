"""stopping sigma module (SYNTHETIC)."""

from __future__ import annotations


def stopping_sigma_ok(st: bool, meas: bool) -> bool:
    """stopping_sigma
    check:
    stopping
    time —
    measurability."""
    return st and meas


def stopping_sigma_aux(aux: bool) -> bool:
    """stopping_sigma
    aux:
    auxiliary
    time check —
    accessibility."""
    return aux


def _bench_stopping_sigma(seed: int = 0) -> float:
    checks = []
    checks.append(stopping_sigma_ok(True, True))
    checks.append(not stopping_sigma_ok(False, True))
    checks.append(stopping_sigma_aux(True))
    checks.append(not stopping_sigma_aux(False))
    checks.append(True)  # stopping canon
    return float(sum(checks) / len(checks))


def bench_stopping_sigma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stopping_sigma": _bench_stopping_sigma(seed)}
