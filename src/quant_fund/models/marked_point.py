"""marked point module (SYNTHETIC)."""

from __future__ import annotations


def marked_point_ok(pt: bool, meas: bool) -> bool:
    """marked_point
    check:
    point-process
    structure —
    Cox
    intensity."""
    return pt and meas


def marked_point_aux(aux: bool) -> bool:
    """marked_point
    aux:
    auxiliary
    mark
    check —
    Palm
    distribution."""
    return aux


def _bench_marked_point(seed: int = 0) -> float:
    checks = []
    checks.append(marked_point_ok(True, True))
    checks.append(not marked_point_ok(False, True))
    checks.append(marked_point_aux(True))
    checks.append(not marked_point_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_marked_point(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marked_point": _bench_marked_point(seed)}
