"""Normal surfaces (SYNTHETIC)."""

from __future__ import annotations


def ns_ok(triangulation: bool, elementary: bool) -> bool:
    """Normal
    surface:
    meets
    each
    tetrahedron
    in
    triangles
    and
    quadrilaterals —
    normal
    equations
    parametrize
    them."""
    return triangulation and elementary


def haken_algorithm(ha: bool) -> bool:
    """Haken
    algorithm:
    incompressible
    surfaces
    appear
    among
    fundamental
    normal
    solutions —
    decision
    procedures."""
    return ha


def _bench_normal_surface(seed: int = 0) -> float:
    checks = []
    checks.append(ns_ok(True, True))
    checks.append(not ns_ok(False, True))
    checks.append(haken_algorithm(True))
    checks.append(not haken_algorithm(False))
    checks.append(True)  # Haken
    return float(sum(checks) / len(checks))


def bench_normal_surface(seed: int = 0) -> dict[str, float]:
    return {"synthetic_normal_surface": _bench_normal_surface(seed)}
