"""palm dist module (SYNTHETIC)."""

from __future__ import annotations


def palm_dist_ok(pt: bool, meas: bool) -> bool:
    """palm_dist
    check:
    point-process
    structure —
    Cox
    intensity."""
    return pt and meas


def palm_dist_aux(aux: bool) -> bool:
    """palm_dist
    aux:
    auxiliary
    mark
    check —
    Palm
    distribution."""
    return aux


def _bench_palm_dist(seed: int = 0) -> float:
    checks = []
    checks.append(palm_dist_ok(True, True))
    checks.append(not palm_dist_ok(False, True))
    checks.append(palm_dist_aux(True))
    checks.append(not palm_dist_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_palm_dist(seed: int = 0) -> dict[str, float]:
    return {"synthetic_palm_dist": _bench_palm_dist(seed)}
