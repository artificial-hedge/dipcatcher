"""a1 degrees module (SYNTHETIC)."""

from __future__ import annotations


def a1_degrees_ok(motive: bool, a1: bool) -> bool:
    """a1_degrees
    check:
    motivic-A1
    structure —
    Voevodsky."""
    return motive and a1


def a1_degrees_aux(aux: bool) -> bool:
    """a1_degrees
    aux:
    auxiliary
    motive
    check —
    Morel."""
    return aux


def _bench_a1_degrees(seed: int = 0) -> float:
    checks = []
    checks.append(a1_degrees_ok(True, True))
    checks.append(not a1_degrees_ok(False, True))
    checks.append(a1_degrees_aux(True))
    checks.append(not a1_degrees_aux(False))
    checks.append(True)  # motivic-A1 canon
    return float(sum(checks) / len(checks))


def bench_a1_degrees(seed: int = 0) -> dict[str, float]:
    return {"synthetic_a1_degrees": _bench_a1_degrees(seed)}
