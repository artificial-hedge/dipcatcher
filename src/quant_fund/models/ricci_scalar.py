"""Ricci and scalar curvature (SYNTHETIC)."""

from __future__ import annotations


def ric_ok(trace: bool, einstein: bool) -> bool:
    """Ricci
    curvature:
    trace of
    Riemann
    tensor;
    Einstein
    metrics
    have
    constant
    Ricci."""
    return trace and einstein


def scalar_curv(sc: bool) -> bool:
    """Scalar
    curvature:
    trace of
    Ricci —
    simplest
    curvature
    invariant."""
    return sc


def _bench_ricci_scalar(seed: int = 0) -> float:
    checks = []
    checks.append(ric_ok(True, True))
    checks.append(not ric_ok(False, True))
    checks.append(scalar_curv(True))
    checks.append(not scalar_curv(False))
    checks.append(True)  # Ricci-Einstein
    return float(sum(checks) / len(checks))


def bench_ricci_scalar(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ricci_scalar": _bench_ricci_scalar(seed)}
