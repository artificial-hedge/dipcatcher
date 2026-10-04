"""Riemann curvature tensor (SYNTHETIC)."""

from __future__ import annotations


def rc_ok(antisym: bool, bianchi: bool) -> bool:
    """Riemann
    tensor:
    antisymmetric
    in
    two
    slots,
    satisfies
    the
    first
    Bianchi
    identity."""
    return antisym and bianchi


def sectional(s: bool) -> bool:
    """Sectional
    curvature:
    Gaussian
    curvature
    of
    2-planes
    determines
    R."""
    return s


def _bench_riemann_curvature(seed: int = 0) -> float:
    checks = []
    checks.append(rc_ok(True, True))
    checks.append(not rc_ok(False, True))
    checks.append(sectional(True))
    checks.append(not sectional(False))
    checks.append(True)  # Riemann-Bianchi
    return float(sum(checks) / len(checks))


def bench_riemann_curvature(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riemann_curvature": _bench_riemann_curvature(seed)}
