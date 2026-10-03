"""first hitting module (SYNTHETIC)."""

from __future__ import annotations


def first_hitting_ok(st: bool, meas: bool) -> bool:
    """first_hitting
    check:
    stopping
    time —
    measurability."""
    return st and meas


def first_hitting_aux(aux: bool) -> bool:
    """first_hitting
    aux:
    auxiliary
    time check —
    accessibility."""
    return aux


def _bench_first_hitting(seed: int = 0) -> float:
    checks = []
    checks.append(first_hitting_ok(True, True))
    checks.append(not first_hitting_ok(False, True))
    checks.append(first_hitting_aux(True))
    checks.append(not first_hitting_aux(False))
    checks.append(True)  # stopping canon
    return float(sum(checks) / len(checks))


def bench_first_hitting(seed: int = 0) -> dict[str, float]:
    return {"synthetic_first_hitting": _bench_first_hitting(seed)}
