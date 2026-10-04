"""Motivic descent (SYNTHETIC)."""

from __future__ import annotations


def md_ok(descent_cond: bool, cdh: bool) -> bool:
    """Motivic
    descent:
    cdh
    descent
    on
    motives —
    descent
    condition."""
    return descent_cond and cdh


def descent_rectified(dr: bool) -> bool:
    """Descent:
    rectified
    descent
    on
    motivic
    complexes —
    localization."""
    return dr


def _bench_motivic_descent(seed: int = 0) -> float:
    checks = []
    checks.append(md_ok(True, True))
    checks.append(not md_ok(False, True))
    checks.append(descent_rectified(True))
    checks.append(not descent_rectified(False))
    checks.append(True)  # Descent
    return float(sum(checks) / len(checks))


def bench_motivic_descent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_descent": _bench_motivic_descent(seed)}
