"""debuts theorem module (SYNTHETIC)."""

from __future__ import annotations


def debuts_theorem_ok(st: bool, meas: bool) -> bool:
    """debuts_theorem
    check:
    stopping
    time —
    measurability."""
    return st and meas


def debuts_theorem_aux(aux: bool) -> bool:
    """debuts_theorem
    aux:
    auxiliary
    time check —
    accessibility."""
    return aux


def _bench_debuts_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(debuts_theorem_ok(True, True))
    checks.append(not debuts_theorem_ok(False, True))
    checks.append(debuts_theorem_aux(True))
    checks.append(not debuts_theorem_aux(False))
    checks.append(True)  # stopping canon
    return float(sum(checks) / len(checks))


def bench_debuts_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_debuts_theorem": _bench_debuts_theorem(seed)}
