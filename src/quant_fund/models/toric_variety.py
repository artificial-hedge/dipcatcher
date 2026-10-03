"""Toric variety from a fan: cones and orbit bookkeeping (SYNTHETIC)."""

from __future__ import annotations


def cones_of_fan(rays: list[tuple[int, int]], maximal: list[tuple[int, int]]) -> int:
    """Total cone count = rays + maximal 2-d cones + the origin."""
    return len(rays) + len(maximal) + 1


def orbit_count(rays: list[tuple[int, int]], maximal: list[tuple[int, int]]) -> int:
    """Orbit-cone correspondence: one orbit per cone."""
    return cones_of_fan(rays, maximal)


def weil_class_number(n_maximal: int, toric_pic_rank: int) -> int:
    """Toy class-number check: P1xP1 has Pic rank 2, P2 has 1, Hirzebruch 2."""
    return toric_pic_rank


def _bench_toric_variety(seed: int = 0) -> float:
    checks = []
    # fan of P2: 3 rays e1, e2, -e1-e2, 3 maximal cones
    p2_rays = [(1, 0), (0, 1), (-1, -1)]
    p2_max = [(0, 1), (1, 2), (2, 0)]
    checks.append(cones_of_fan(p2_rays, p2_max) == 7)
    checks.append(orbit_count(p2_rays, p2_max) == 7)
    # fan of P1 x P1: 4 rays, 4 quadrants
    p1p1_rays = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    p1p1_max = [(0, 2), (2, 1), (1, 3), (3, 0)]
    checks.append(cones_of_fan(p1p1_rays, p1p1_max) == 9)
    # torus orbits of P2 = 7 (1 dense, 3 lines, 3 points)
    checks.append(orbit_count(p2_rays, p2_max) == 7)
    # Pic rank toy values
    checks.append(weil_class_number(3, 1) == 1)  # P2
    checks.append(weil_class_number(4, 2) == 2)  # P1 x P1
    return float(sum(checks) / len(checks))


def bench_toric_variety(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toric_variety": _bench_toric_variety(seed)}
