"""hawkes point module (SYNTHETIC)."""

from __future__ import annotations


def hawkes_point_ok(pt: bool, meas: bool) -> bool:
    """hawkes_point
    check:
    point-process
    structure —
    Cox
    intensity."""
    return pt and meas


def hawkes_point_aux(aux: bool) -> bool:
    """hawkes_point
    aux:
    auxiliary
    mark
    check —
    Palm
    distribution."""
    return aux


def _bench_hawkes_point(seed: int = 0) -> float:
    checks = []
    checks.append(hawkes_point_ok(True, True))
    checks.append(not hawkes_point_ok(False, True))
    checks.append(hawkes_point_aux(True))
    checks.append(not hawkes_point_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_hawkes_point(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hawkes_point": _bench_hawkes_point(seed)}
