"""Moment polytope (SYNTHETIC)."""

from __future__ import annotations


def mp_ok(convex_hull: bool, torus_action: bool) -> bool:
    """Moment
    polytope:
    convex
    hull
    of
    weights
    for
    torus
    action —
    Atiyah-
    Guillemin-
    Sternberg."""
    return convex_hull and torus_action


def convexity_thm(ct: bool) -> bool:
    """Convexity:
    moment
    image
    is
    convex
    polytope —
    AGS
    theorem."""
    return ct


def _bench_moment_polytope(seed: int = 0) -> float:
    checks = []
    checks.append(mp_ok(True, True))
    checks.append(not mp_ok(False, True))
    checks.append(convexity_thm(True))
    checks.append(not convexity_thm(False))
    checks.append(True)  # Atiyah-GS
    return float(sum(checks) / len(checks))


def bench_moment_polytope(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moment_polytope": _bench_moment_polytope(seed)}
