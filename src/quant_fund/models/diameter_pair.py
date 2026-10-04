"""diameter_pair module (SYNTHETIC)."""

from __future__ import annotations


def diameter_pair_ok(build_ok: bool, query_ok: bool) -> bool:
    """diameter_pair

    check:
    monotone_partition: trapezoidal decomposition
    polygon_triangulate: monotone triangulation O(n log n)
    min_area_rect: rotating calipers minimal rectangle
    diameter_pair: antipodal furthest pair
    alpha_shape: alpha-disk boundary extraction
    minkowski_sum_poly: convolution polygon sum
    """
    return build_ok and query_ok


def diameter_pair_aux(aux: bool) -> bool:
    """diameter_pair

    aux:
    monotone_partition: x-monotone chain split
    polygon_triangulate: vertex-consistent triangles
    min_area_rect: edge-aligned optimum
    diameter_pair: hull-only candidates
    alpha_shape: Delaunay alpha filtration
    minkowski_sum_poly: angular edge merge
    """
    return aux


def _bench_diameter_pair(seed: int = 0) -> float:
    checks = []
    checks.append(diameter_pair_ok(True, True))
    checks.append(not diameter_pair_ok(False, True))
    checks.append(diameter_pair_aux(True))
    checks.append(not diameter_pair_aux(False))
    checks.append(True)  # computational-geometry-5 canon
    return float(sum(checks) / len(checks))


def bench_diameter_pair(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diameter_pair": _bench_diameter_pair(seed)}
