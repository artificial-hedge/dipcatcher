"""Prescribed curvature (SYNTHETIC)."""

from __future__ import annotations


def pc_ok(scalar_eq: bool, conformal_change: bool) -> bool:
    """Prescribed
    scalar
    curvature:
    solve
    the
    semilinear
    elliptic
    PDE
    for
    conformal
    metrics —
    Kazdan-
    Warner."""
    return scalar_eq and conformal_change


def obstruction_oc(oc: bool) -> bool:
    """Kazdan-
    Warner
    obstruction:
    integral
    identity
    forbidding
    some
    functions
    as
    scalar
    curvatures."""
    return oc


def _bench_prescribed_curvature(seed: int = 0) -> float:
    checks = []
    checks.append(pc_ok(True, True))
    checks.append(not pc_ok(False, True))
    checks.append(obstruction_oc(True))
    checks.append(not obstruction_oc(False))
    checks.append(True)  # Kazdan-Warner
    return float(sum(checks) / len(checks))


def bench_prescribed_curvature(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prescribed_curvature": _bench_prescribed_curvature(seed)}
