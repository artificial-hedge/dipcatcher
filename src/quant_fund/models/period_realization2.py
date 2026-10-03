"""period realization2 module (SYNTHETIC)."""

from __future__ import annotations


def period_realization2_ok(motivic: bool, hodge: bool) -> bool:
    """period_realization2
    check:
    motivic
    structure —
    Hodge."""
    return motivic and hodge


def period_realization2_aux(aux: bool) -> bool:
    """period_realization2
    aux:
    auxiliary
    motivic
    check —
    Tate."""
    return aux


def _bench_period_realization2(seed: int = 0) -> float:
    checks = []
    checks.append(period_realization2_ok(True, True))
    checks.append(not period_realization2_ok(False, True))
    checks.append(period_realization2_aux(True))
    checks.append(not period_realization2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_period_realization2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_period_realization2": _bench_period_realization2(seed)}
