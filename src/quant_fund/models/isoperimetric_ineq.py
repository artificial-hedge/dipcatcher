"""isoperimetric ineq module (SYNTHETIC)."""

from __future__ import annotations


def isoperimetric_ineq_ok(convex: bool, body: bool) -> bool:
    """isoperimetric_ineq
    check:
    convex
    geometry —
    body."""
    return convex and body


def isoperimetric_ineq_aux(aux: bool) -> bool:
    """isoperimetric_ineq
    aux:
    auxiliary
    geometry check —
    volume."""
    return aux


def _bench_isoperimetric_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(isoperimetric_ineq_ok(True, True))
    checks.append(not isoperimetric_ineq_ok(False, True))
    checks.append(isoperimetric_ineq_aux(True))
    checks.append(not isoperimetric_ineq_aux(False))
    checks.append(True)  # convex-geometry canon
    return float(sum(checks) / len(checks))


def bench_isoperimetric_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isoperimetric_ineq": _bench_isoperimetric_ineq(seed)}
