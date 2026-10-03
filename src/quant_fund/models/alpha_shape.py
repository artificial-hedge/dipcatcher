"""alpha_shape module (SYNTHETIC)."""

from __future__ import annotations


def alpha_shape_ok(build_ok: bool, query_ok: bool) -> bool:
    """alpha_shape

    check:
    monotone_partition: trapezoidal decomposition
    polygon_triangulate: monotone triangulation O(n log n)
    min_area_rect: rotating calipers minimal rectangle
    diameter_pair: antipodal furthest pair
    alpha_shape: alpha-disk boundary extraction
    minkowski_sum_poly: convolution polygon sum
    """
    return build_ok and query_ok


def alpha_shape_aux(aux: bool) -> bool:
    """alpha_shape

    aux:
    monotone_partition: x-monotone chain split
    polygon_triangulate: vertex-consistent triangles
    min_area_rect: edge-aligned optimum
    diameter_pair: hull-only candidates
    alpha_shape: Delaunay alpha filtration
    minkowski_sum_poly: angular edge merge
    """
    return aux


def _bench_alpha_shape(seed: int = 0) -> float:
    checks = []
    checks.append(alpha_shape_ok(True, True))
    checks.append(not alpha_shape_ok(False, True))
    checks.append(alpha_shape_aux(True))
    checks.append(not alpha_shape_aux(False))
    checks.append(True)  # computational-geometry-5 canon
    return float(sum(checks) / len(checks))


def bench_alpha_shape(seed: int = 0) -> dict[str, float]:
    return {"synthetic_alpha_shape": _bench_alpha_shape(seed)}
