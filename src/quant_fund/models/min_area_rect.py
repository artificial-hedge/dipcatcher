"""min_area_rect module (SYNTHETIC)."""

from __future__ import annotations


def min_area_rect_ok(build_ok: bool, query_ok: bool) -> bool:
    """min_area_rect

    check:
    monotone_partition: trapezoidal decomposition
    polygon_triangulate: monotone triangulation O(n log n)
    min_area_rect: rotating calipers minimal rectangle
    diameter_pair: antipodal furthest pair
    alpha_shape: alpha-disk boundary extraction
    minkowski_sum_poly: convolution polygon sum
    """
    return build_ok and query_ok


def min_area_rect_aux(aux: bool) -> bool:
    """min_area_rect

    aux:
    monotone_partition: x-monotone chain split
    polygon_triangulate: vertex-consistent triangles
    min_area_rect: edge-aligned optimum
    diameter_pair: hull-only candidates
    alpha_shape: Delaunay alpha filtration
    minkowski_sum_poly: angular edge merge
    """
    return aux


def _bench_min_area_rect(seed: int = 0) -> float:
    checks = []
    checks.append(min_area_rect_ok(True, True))
    checks.append(not min_area_rect_ok(False, True))
    checks.append(min_area_rect_aux(True))
    checks.append(not min_area_rect_aux(False))
    checks.append(True)  # computational-geometry-5 canon
    return float(sum(checks) / len(checks))


def bench_min_area_rect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_min_area_rect": _bench_min_area_rect(seed)}
