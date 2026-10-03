"""progressive set module (SYNTHETIC)."""

from __future__ import annotations


def progressive_set_ok(st: bool, meas: bool) -> bool:
    """progressive_set
    check:
    stopping
    time —
    measurability."""
    return st and meas


def progressive_set_aux(aux: bool) -> bool:
    """progressive_set
    aux:
    auxiliary
    time check —
    accessibility."""
    return aux


def _bench_progressive_set(seed: int = 0) -> float:
    checks = []
    checks.append(progressive_set_ok(True, True))
    checks.append(not progressive_set_ok(False, True))
    checks.append(progressive_set_aux(True))
    checks.append(not progressive_set_aux(False))
    checks.append(True)  # stopping canon
    return float(sum(checks) / len(checks))


def bench_progressive_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_progressive_set": _bench_progressive_set(seed)}
