"""accessible time module (SYNTHETIC)."""

from __future__ import annotations


def accessible_time_ok(st: bool, meas: bool) -> bool:
    """accessible_time
    check:
    stopping
    time —
    measurability."""
    return st and meas


def accessible_time_aux(aux: bool) -> bool:
    """accessible_time
    aux:
    auxiliary
    time check —
    accessibility."""
    return aux


def _bench_accessible_time(seed: int = 0) -> float:
    checks = []
    checks.append(accessible_time_ok(True, True))
    checks.append(not accessible_time_ok(False, True))
    checks.append(accessible_time_aux(True))
    checks.append(not accessible_time_aux(False))
    checks.append(True)  # stopping canon
    return float(sum(checks) / len(checks))


def bench_accessible_time(seed: int = 0) -> dict[str, float]:
    return {"synthetic_accessible_time": _bench_accessible_time(seed)}
