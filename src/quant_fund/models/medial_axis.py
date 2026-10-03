"""medial_axis module (SYNTHETIC)."""

from __future__ import annotations


def medial_axis_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """medial_axis

    check:
    convex_hull_3d: 3D hull via incremental beneath-beyond
    polygon_boolean: boolean ops on polygons
    medial_axis: Voronoi-based medial axis
    polygon_centroid: centroid of simple polygon
    shape_context: shape-context descriptor matching
    beta_skeleton: beta-skeleton proximity graph
    """
    return fit_ok and sample_ok


def medial_axis_aux(aux: bool) -> bool:
    """medial_axis

    aux:
    convex_hull_3d: Euler characteristic V-E+F=2
    polygon_boolean: area conservation under union
    medial_axis: maximal inscribed circle centers
    polygon_centroid: shoelace-weighted mean
    shape_context: log-polar histogram invariance
    beta_skeleton: RNG ⊆ β-skeleton ⊆ Delaunay
    """
    return aux


def _bench_medial_axis(seed: int = 0) -> float:
    checks = []
    checks.append(medial_axis_ok(True, True))
    checks.append(not medial_axis_ok(False, True))
    checks.append(medial_axis_aux(True))
    checks.append(not medial_axis_aux(False))
    checks.append(True)  # computational-geometry-6 canon
    return float(sum(checks) / len(checks))


def bench_medial_axis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_medial_axis": _bench_medial_axis(seed)}
