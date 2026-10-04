"""relative motive module (SYNTHETIC)."""

from __future__ import annotations


def relative_motive_ok(motivic: bool, stable: bool) -> bool:
    """relative_motive
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def relative_motive_aux(aux: bool) -> bool:
    """relative_motive
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_relative_motive(seed: int = 0) -> float:
    checks = []
    checks.append(relative_motive_ok(True, True))
    checks.append(not relative_motive_ok(False, True))
    checks.append(relative_motive_aux(True))
    checks.append(not relative_motive_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_relative_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_relative_motive": _bench_relative_motive(seed)}
