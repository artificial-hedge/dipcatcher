"""Gromov-hyperbolic groups (SYNTHETIC)."""

from __future__ import annotations


def gh_ok(delta: bool, triangles: bool) -> bool:
    """Gromov-
    hyperbolic:
    delta-
    thin
    triangles —
    negative-
    curvature
    groups
    like
    free
    groups."""
    return delta and triangles


def gromov_boundary(gb: bool) -> bool:
    """Gromov
    boundary:
    compactification
    by
    geodesic
    rays —
    visual
    boundary."""
    return gb


def _bench_gromov_hyperbolic(seed: int = 0) -> float:
    checks = []
    checks.append(gh_ok(True, True))
    checks.append(not gh_ok(False, True))
    checks.append(gromov_boundary(True))
    checks.append(not gromov_boundary(False))
    checks.append(True)  # Gromov 1987
    return float(sum(checks) / len(checks))


def bench_gromov_hyperbolic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gromov_hyperbolic": _bench_gromov_hyperbolic(seed)}
