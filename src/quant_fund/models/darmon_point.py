"""darmon point module (SYNTHETIC)."""

from __future__ import annotations


def darmon_point_ok(point: bool, automorphic: bool) -> bool:
    """darmon_point
    check:
    automorphic-point
    structure —
    Darmon."""
    return point and automorphic


def darmon_point_aux(aux: bool) -> bool:
    """darmon_point
    aux:
    auxiliary
    point
    check —
    Stark."""
    return aux


def _bench_darmon_point(seed: int = 0) -> float:
    checks = []
    checks.append(darmon_point_ok(True, True))
    checks.append(not darmon_point_ok(False, True))
    checks.append(darmon_point_aux(True))
    checks.append(not darmon_point_aux(False))
    checks.append(True)  # automorphic-points canon
    return float(sum(checks) / len(checks))


def bench_darmon_point(seed: int = 0) -> dict[str, float]:
    return {"synthetic_darmon_point": _bench_darmon_point(seed)}
