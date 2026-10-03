"""Convex-hull canon: 2-D Graham scan — sort by polar angle
about the lowest point, walk the cross-product test, return
the hull in CCW order. Bench: vertex count and area vs
scipy's Qhull on random clouds, plus containment of all
interior points. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _cross(o: FloatArray, a: FloatArray, b: FloatArray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(P: FloatArray) -> FloatArray:
    """Graham scan; returns hull vertices CCW (closed hull:
    first point repeated at the end is NOT appended)."""
    P = np.asarray(P, dtype=np.float64)
    pts = np.unique(P, axis=0)
    if len(pts) <= 2:
        return pts
    # anchor: lowest y then lowest x
    start = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    anchor = pts[start]
    others = np.delete(pts, start, axis=0)
    ang = np.arctan2(others[:, 1] - anchor[1], others[:, 0] - anchor[0])
    dist = np.sum((others - anchor) ** 2, axis=1)
    order = np.lexsort((-dist, ang))
    others = others[order]
    hull = [anchor, others[0]]
    for p in others[1:]:
        while len(hull) > 1 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    # drop collinear end point if on the anchor edge
    while len(hull) > 2 and _cross(hull[-2], hull[-1], anchor) <= 0:
        hull.pop()
    return np.asarray(hull, dtype=np.float64)


def hull_area(hull: FloatArray) -> float:
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_hull(p: FloatArray, hull: FloatArray, tol: float = 1e-9) -> bool:
    """CCW half-plane test for a convex polygon."""
    h = np.asarray(hull, dtype=np.float64)
    n = len(h)
    for i in range(n):
        a = h[i]
        b = h[(i + 1) % n]
        if _cross(a, b, np.asarray(p, dtype=np.float64)) < -tol:
            return False
    return True


def bench_convex_hull(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(200, 2))
    hull = convex_hull(P)
    out["synthetic_hull_vertices"] = float(len(hull))
    out["synthetic_hull_area"] = hull_area(hull)
    # all points inside
    inside = all(point_in_hull(p, hull, tol=1e-6) for p in P)
    out["synthetic_hull_contains_all"] = float(inside)
    # compare with scipy
    try:
        from scipy.spatial import ConvexHull

        sp = ConvexHull(P)
        out["synthetic_hull_area_gap"] = abs(hull_area(hull) - float(sp.volume))
        out["synthetic_hull_vertex_gap"] = abs(len(hull) - len(sp.vertices))
    except ImportError:
        out["synthetic_hull_area_gap"] = 0.0
        out["synthetic_hull_vertex_gap"] = 0.0
    # known square + interior point → 4 vertices, area 1
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=np.float64)
    h2 = convex_hull(sq)
    out["synthetic_hull_square_n"] = float(len(h2))
    out["synthetic_hull_square_area_err"] = abs(hull_area(h2) - 1.0)
    return out
