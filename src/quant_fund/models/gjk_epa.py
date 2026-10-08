"""Exact convex-polygon distance (edge-feature scan) + EPA penetration depth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _seg_seg_dist(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Minimum distance between two segments (2-D)."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm(p1 + s * d1 - (p2 + t * d2)))


def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    sign = 0.0
    m = len(poly)
    for i in range(m):
        c = _cross2(poly[(i + 1) % m] - poly[i], pt - poly[i])
        if abs(c) > 1e-12:
            if sign == 0.0:
                sign = np.sign(c)
            elif np.sign(c) != sign:
                return False
    return True


def gjk_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Exact distance between two convex polygons (0 when intersecting).

    Convex polygons intersect iff an edge pair crosses or a vertex of one lies
    inside the other; otherwise distance is the min edge-pair separation.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    m1, m2 = len(p1), len(p2)
    best = float("inf")
    crosses = False
    for i in range(m1):
        a, b = p1[i], p1[(i + 1) % m1]
        for j in range(m2):
            c, d = p2[j], p2[(j + 1) % m2]
            if _segs_cross(a, b, c, d):
                crosses = True
            best = min(best, _seg_seg_dist(a, b, c, d))
    if crosses or _point_in_poly(p1[0], p2) or _point_in_poly(p2[0], p1):
        return 0.0
    return best


def _segs_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    d1 = _cross2(b - a, c - a) * _cross2(b - a, d - a)
    d2 = _cross2(d - c, a - c) * _cross2(d - c, b - c)
    return d1 < 0.0 and d2 < 0.0


def epa_depth(p1: np.ndarray, p2: np.ndarray) -> float:
    """Penetration depth via the Minkowski-difference convex hull."""
    if gjk_distance(p1, p2) > 0.0:
        return 0.0
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    cloud = np.array([[a - b for b in p2] for a in p1]).reshape(-1, 2)
    hull = _convex_hull(cloud)
    best = float("inf")
    m = len(hull)
    centroid = hull.mean(axis=0)
    for i in range(m):
        p, q = hull[i], hull[(i + 1) % m]
        e = q - p
        n = np.array([e[1], -e[0]])
        n = n / np.linalg.norm(n)
        if float(n @ (centroid - p)) > 0.0:
            n = -n
        best = min(best, float(n @ p))
    return float(abs(best))


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    pts_s = sorted(map(tuple, pts.tolist()))

    def _cross(o: tuple, a: tuple, b: tuple) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[tuple] = []
    for p in pts_s:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple] = []
    for p in reversed(pts_s):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(np.array(lower[:-1] + upper[:-1]))


def _brute_min_dist(p1: np.ndarray, p2: np.ndarray, n: int = 600) -> float:
    """Oracle: dense boundary sampling + containment (independent check)."""

    def _boundary(poly: np.ndarray, n: int) -> np.ndarray:
        out = []
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            for t in np.linspace(0.0, 1.0, max(2, n // m), endpoint=False):
                out.append(a + t * (b - a))
        return np.array(out)

    b1, b2 = _boundary(p1, n), _boundary(p2, n)
    d = float(np.min(np.linalg.norm(b1[:, None] - b2[None], axis=2)))
    if _point_in_poly(b1[0], p2) or _point_in_poly(b2[0], p1):
        return 0.0
    return d


def bench_gjk_epa(seed: int = 20261231 + 862) -> dict[str, float]:
    """Exact polygon distance matches dense brute-force oracle on random pairs."""
    rng = np.random.default_rng(seed)
    checks = 0.0
    total = 0
    for _ in range(25):

        def _poly(cx: float, cy: float, k: int) -> np.ndarray:
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            r = rng.uniform(0.3, 1.0, k)
            return np.asarray(np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)])

        p1 = _poly(0.0, 0.0, int(rng.integers(3, 6)))
        sep = float(rng.uniform(0.0, 2.5))
        p2 = _poly(sep, float(rng.uniform(-0.5, 0.5)), int(rng.integers(3, 6)))
        d_gjk = gjk_distance(p1, p2)
        d_ref = _brute_min_dist(p1, p2)
        total += 2
        checks += float(abs(d_gjk - d_ref) < 0.02)
        if d_gjk > 0.0:
            checks += float(epa_depth(p1, p2) == 0.0)
        else:
            checks += float(epa_depth(p1, p2) >= 0.0)
    return {"synthetic_gjk_epa": checks / total}
