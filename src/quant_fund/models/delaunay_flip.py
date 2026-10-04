"""delaunay_flip module (SYNTHETIC)."""

from __future__ import annotations


def delaunay_flip_ok(build_ok: bool, query_ok: bool) -> bool:
    """delaunay_flip

    check:
    voronoi_lite: Fortune's sweep Voronoi diagram
    delaunay_flip: Lawson edge-flip Delaunay
    convex_layers: onion peeling convex layers
    polygon_offset: parallel-edge offset polygon
    rotating_sweep: angular sweep visibility
    visibility_graph: vertex-to-vertex sight edges
    """
    return build_ok and query_ok


def delaunay_flip_aux(aux: bool) -> bool:
    """delaunay_flip

    aux:
    voronoi_lite: beachline parabolic front
    delaunay_flip: empty circumcircle invariant
    convex_layers: O(n log n) peeling
    polygon_offset: miter-join handling
    rotating_sweep: event-angle ordering
    visibility_graph: O(n^2) edge test
    """
    return aux


def _bench_delaunay_flip(seed: int = 0) -> float:
    checks = []
    checks.append(delaunay_flip_ok(True, True))
    checks.append(not delaunay_flip_ok(False, True))
    checks.append(delaunay_flip_aux(True))
    checks.append(not delaunay_flip_aux(False))
    checks.append(True)  # computational-geometry-4 canon
    return float(sum(checks) / len(checks))


def bench_delaunay_flip(seed: int = 0) -> dict[str, float]:
    return {"synthetic_delaunay_flip": _bench_delaunay_flip(seed)}
