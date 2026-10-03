"""Delaunay canon: Bowyer–Watson incremental triangulation —
super-triangle seed, circumcircle test per insertion, bad-triangle
cavity retriangulation. Bench: empty-circumcircle property on
random clouds and triangle agreement vs scipy's Qhull. All
SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _circum_contains(a: FloatArray, b: FloatArray, c: FloatArray, p: FloatArray) -> bool:
    """In-circle test: is p strictly inside the circumcircle of
    triangle (a,b,c)? Uses the determinant predicate."""
    ax, ay = a[0] - p[0], a[1] - p[1]
    bx, by = b[0] - p[0], b[1] - p[1]
    cx, cy = c[0] - p[0], c[1] - p[1]
    det = (
        (ax * ax + ay * ay) * (bx * cy - cx * by)
        - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
    )
    # orientation of the triangle matters: for CCW triangle,
    # det>0 ⟺ inside
    orient = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return bool(det > 1e-12) if orient > 0 else bool(det < -1e-12)


def delaunay(P: FloatArray) -> FloatArray:
    """Bowyer–Watson triangulation; returns (m,3) vertex-index
    simplices."""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    # super-triangle enclosing all points
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    span = float(maxs.max() - mins.min()) + 1.0
    mid = (mins + maxs) / 2
    super_pts = np.array(
        [
            [mid[0] - 20 * span, mid[1] - span],
            [mid[0], mid[1] + 20 * span],
            [mid[0] + 20 * span, mid[1] - span],
        ]
    )
    pts = np.vstack([P, super_pts])
    tris = [(n, n + 1, n + 2)]
    for i in range(n):
        p = pts[i]
        bad = [t for t in tris if _circum_contains(pts[t[0]], pts[t[1]], pts[t[2]], p)]
        # cavity boundary: edges appearing in exactly one bad tri
        edge_count: dict[tuple[int, int], int] = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                e2 = (min(e), max(e))
                edge_count[e2] = edge_count.get(e2, 0) + 1
        boundary = [e for e, c in edge_count.items() if c == 1]
        bad_set = set(bad)
        tris = [t for t in tris if t not in bad_set]
        for e in boundary:
            tris.append((e[0], e[1], i))
    # drop triangles touching the super vertices
    out = [t for t in tris if all(v < n for v in t)]
    return np.asarray(out, dtype=np.int64)


def _empty_circum_property(P: FloatArray, tris: FloatArray, samples: int = 400) -> float:
    """Max number of points strictly inside any circumcircle —
    should be 0 for a Delaunay triangulation."""
    worst = 0
    for t in tris[:samples]:
        a, b, c = P[t[0]], P[t[1]], P[t[2]]
        cnt = sum(1 for p in P if _circum_contains(a, b, c, p))
        worst = max(worst, cnt)
    return float(worst)


def bench_delaunay(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, 1, (60, 2))
    tris = delaunay(P)
    out["synthetic_delaunay_triangles"] = float(len(tris))
    out["synthetic_delaunay_empty_viol"] = _empty_circum_property(P, tris)
    # agreement with scipy
    try:
        from scipy.spatial import Delaunay

        sp = Delaunay(P)
        mine = {tuple(sorted(t)) for t in tris}
        theirs = {tuple(sorted(t)) for t in sp.simplices}
        out["synthetic_delaunay_agree"] = float(len(mine & theirs))
        out["synthetic_delaunay_scipy_n"] = float(len(theirs))
    except ImportError:
        out["synthetic_delaunay_agree"] = 0.0
        out["synthetic_delaunay_scipy_n"] = 0.0
    # grid sanity: 3x3 grid → 8 triangles
    g = np.array([[i, j] for i in range(3) for j in range(3)], dtype=np.float64)
    tg = delaunay(g)
    out["synthetic_delaunay_grid_n"] = float(len(tg))

    # total triangle area ≈ hull area of the square
    def _area(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2

    tot = sum(_area(g[t[0]], g[t[1]], g[t[2]]) for t in tg)
    out["synthetic_delaunay_grid_area_err"] = abs(tot - 4.0)
    return out
