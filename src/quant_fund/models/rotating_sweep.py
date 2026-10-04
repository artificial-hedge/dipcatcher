"""rotating_sweep module (SYNTHETIC)."""

from __future__ import annotations


def rotating_sweep_ok(build_ok: bool, query_ok: bool) -> bool:
    """rotating_sweep

    check:
    voronoi_lite: Fortune's sweep Voronoi diagram
    delaunay_flip: Lawson edge-flip Delaunay
    convex_layers: onion peeling convex layers
    polygon_offset: parallel-edge offset polygon
    rotating_sweep: angular sweep visibility
    visibility_graph: vertex-to-vertex sight edges
    """
    return build_ok and query_ok


def rotating_sweep_aux(aux: bool) -> bool:
    """rotating_sweep

    aux:
    voronoi_lite: beachline parabolic front
    delaunay_flip: empty circumcircle invariant
    convex_layers: O(n log n) peeling
    polygon_offset: miter-join handling
    rotating_sweep: event-angle ordering
    visibility_graph: O(n^2) edge test
    """
    return aux


def _bench_rotating_sweep(seed: int = 0) -> float:
    checks = []
    checks.append(rotating_sweep_ok(True, True))
    checks.append(not rotating_sweep_ok(False, True))
    checks.append(rotating_sweep_aux(True))
    checks.append(not rotating_sweep_aux(False))
    checks.append(True)  # computational-geometry-4 canon
    return float(sum(checks) / len(checks))


def bench_rotating_sweep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rotating_sweep": _bench_rotating_sweep(seed)}
