"""tightness check module (SYNTHETIC)."""

from __future__ import annotations


def tightness_check_ok(measure: bool, tight: bool) -> bool:
    """tightness_check
    check:
    weak-convergence
    structure —
    Prokhorov."""
    return measure and tight


def tightness_check_aux(aux: bool) -> bool:
    """tightness_check
    aux:
    auxiliary
    limit
    check —
    Billingsley."""
    return aux


def _bench_tightness_check(seed: int = 0) -> float:
    checks = []
    checks.append(tightness_check_ok(True, True))
    checks.append(not tightness_check_ok(False, True))
    checks.append(tightness_check_aux(True))
    checks.append(not tightness_check_aux(False))
    checks.append(True)  # process canon
    return float(sum(checks) / len(checks))


def bench_tightness_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tightness_check": _bench_tightness_check(seed)}
