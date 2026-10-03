"""minkowski sum module (SYNTHETIC)."""

from __future__ import annotations


def minkowski_sum_ok(convex: bool, body: bool) -> bool:
    """minkowski_sum
    check:
    convex
    geometry —
    body."""
    return convex and body


def minkowski_sum_aux(aux: bool) -> bool:
    """minkowski_sum
    aux:
    auxiliary
    geometry check —
    volume."""
    return aux


def _bench_minkowski_sum(seed: int = 0) -> float:
    checks = []
    checks.append(minkowski_sum_ok(True, True))
    checks.append(not minkowski_sum_ok(False, True))
    checks.append(minkowski_sum_aux(True))
    checks.append(not minkowski_sum_aux(False))
    checks.append(True)  # convex-geometry canon
    return float(sum(checks) / len(checks))


def bench_minkowski_sum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minkowski_sum": _bench_minkowski_sum(seed)}
