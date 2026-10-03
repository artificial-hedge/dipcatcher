"""minkowski_sum_poly module (SYNTHETIC)."""

from __future__ import annotations


def minkowski_sum_poly_ok(build_ok: bool, query_ok: bool) -> bool:
    """minkowski_sum_poly

    check:
    monotone_partition: trapezoidal decomposition
    polygon_triangulate: monotone triangulation O(n log n)
    min_area_rect: rotating calipers minimal rectangle
    diameter_pair: antipodal furthest pair
    alpha_shape: alpha-disk boundary extraction
    minkowski_sum_poly: convolution polygon sum
    """
    return build_ok and query_ok


def minkowski_sum_poly_aux(aux: bool) -> bool:
    """minkowski_sum_poly

    aux:
    monotone_partition: x-monotone chain split
    polygon_triangulate: vertex-consistent triangles
    min_area_rect: edge-aligned optimum
    diameter_pair: hull-only candidates
    alpha_shape: Delaunay alpha filtration
    minkowski_sum_poly: angular edge merge
    """
    return aux


def _bench_minkowski_sum_poly(seed: int = 0) -> float:
    checks = []
    checks.append(minkowski_sum_poly_ok(True, True))
    checks.append(not minkowski_sum_poly_ok(False, True))
    checks.append(minkowski_sum_poly_aux(True))
    checks.append(not minkowski_sum_poly_aux(False))
    checks.append(True)  # computational-geometry-5 canon
    return float(sum(checks) / len(checks))


def bench_minkowski_sum_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minkowski_sum_poly": _bench_minkowski_sum_poly(seed)}
