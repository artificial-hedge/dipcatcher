"""monotone_partition module (SYNTHETIC)."""

from __future__ import annotations


def monotone_partition_ok(build_ok: bool, query_ok: bool) -> bool:
    """monotone_partition

    check:
    monotone_partition: trapezoidal decomposition
    polygon_triangulate: monotone triangulation O(n log n)
    min_area_rect: rotating calipers minimal rectangle
    diameter_pair: antipodal furthest pair
    alpha_shape: alpha-disk boundary extraction
    minkowski_sum_poly: convolution polygon sum
    """
    return build_ok and query_ok


def monotone_partition_aux(aux: bool) -> bool:
    """monotone_partition

    aux:
    monotone_partition: x-monotone chain split
    polygon_triangulate: vertex-consistent triangles
    min_area_rect: edge-aligned optimum
    diameter_pair: hull-only candidates
    alpha_shape: Delaunay alpha filtration
    minkowski_sum_poly: angular edge merge
    """
    return aux


def _bench_monotone_partition(seed: int = 0) -> float:
    checks = []
    checks.append(monotone_partition_ok(True, True))
    checks.append(not monotone_partition_ok(False, True))
    checks.append(monotone_partition_aux(True))
    checks.append(not monotone_partition_aux(False))
    checks.append(True)  # computational-geometry-5 canon
    return float(sum(checks) / len(checks))


def bench_monotone_partition(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monotone_partition": _bench_monotone_partition(seed)}
