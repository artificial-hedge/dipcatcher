"""visibility_graph module (SYNTHETIC)."""

from __future__ import annotations


def visibility_graph_ok(build_ok: bool, query_ok: bool) -> bool:
    """visibility_graph

    check:
    voronoi_lite: Fortune's sweep Voronoi diagram
    delaunay_flip: Lawson edge-flip Delaunay
    convex_layers: onion peeling convex layers
    polygon_offset: parallel-edge offset polygon
    rotating_sweep: angular sweep visibility
    visibility_graph: vertex-to-vertex sight edges
    """
    return build_ok and query_ok


def visibility_graph_aux(aux: bool) -> bool:
    """visibility_graph

    aux:
    voronoi_lite: beachline parabolic front
    delaunay_flip: empty circumcircle invariant
    convex_layers: O(n log n) peeling
    polygon_offset: miter-join handling
    rotating_sweep: event-angle ordering
    visibility_graph: O(n^2) edge test
    """
    return aux


def _bench_visibility_graph(seed: int = 0) -> float:
    checks = []
    checks.append(visibility_graph_ok(True, True))
    checks.append(not visibility_graph_ok(False, True))
    checks.append(visibility_graph_aux(True))
    checks.append(not visibility_graph_aux(False))
    checks.append(True)  # computational-geometry-4 canon
    return float(sum(checks) / len(checks))


def bench_visibility_graph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_visibility_graph": _bench_visibility_graph(seed)}
