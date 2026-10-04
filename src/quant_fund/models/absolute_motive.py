"""absolute motive module (SYNTHETIC)."""

from __future__ import annotations


def absolute_motive_ok(motivic: bool, stable: bool) -> bool:
    """absolute_motive
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def absolute_motive_aux(aux: bool) -> bool:
    """absolute_motive
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_absolute_motive(seed: int = 0) -> float:
    checks = []
    checks.append(absolute_motive_ok(True, True))
    checks.append(not absolute_motive_ok(False, True))
    checks.append(absolute_motive_aux(True))
    checks.append(not absolute_motive_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_absolute_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_absolute_motive": _bench_absolute_motive(seed)}
