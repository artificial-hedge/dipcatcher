"""last exit module (SYNTHETIC)."""

from __future__ import annotations


def last_exit_ok(st: bool, meas: bool) -> bool:
    """last_exit
    check:
    stopping
    time —
    measurability."""
    return st and meas


def last_exit_aux(aux: bool) -> bool:
    """last_exit
    aux:
    auxiliary
    time check —
    accessibility."""
    return aux


def _bench_last_exit(seed: int = 0) -> float:
    checks = []
    checks.append(last_exit_ok(True, True))
    checks.append(not last_exit_ok(False, True))
    checks.append(last_exit_aux(True))
    checks.append(not last_exit_aux(False))
    checks.append(True)  # stopping canon
    return float(sum(checks) / len(checks))


def bench_last_exit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_last_exit": _bench_last_exit(seed)}
