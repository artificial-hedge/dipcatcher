"""Ferrari–Canny epsilon grasp-quality metric (2-D, linearized friction cone) (SYNTHETIC).

Each contact contributes a friction-cone wrench set, linearized into L facets.
The union of contact wrench hulls is summed (Minkowski) via enumerating the
extreme rays, then quality = radius of the largest origin-centred ball inside
the convex hull — computed as the min distance from the origin to each hull
edge. Antipodal force-closure grasps score > 0; degenerate ones score 0.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 978


def contact_wrenches(pos: np.ndarray, normal: np.ndarray, mu: float, facets: int = 8) -> np.ndarray:
    t = np.array([-normal[1], normal[0]])
    rays = []
    for k in range(facets):
        a = 2 * np.pi * k / facets
        f = normal + mu * (np.cos(a) * t + np.sin(a) * np.array([t[1], -t[0]]))
        tau = pos[0] * f[1] - pos[1] * f[0]
        rays.append([f[0], f[1], tau])
    return np.asarray(rays)


def convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    p = pts[np.argsort(pts[:, 0] * 1000 + pts[:, 1])]
    hull: list[np.ndarray] = []
    for pt in p:
        while len(hull) >= 2 and np.cross(hull[-1] - hull[-2], np.asarray(pt) - hull[-1]) <= 1e-12:
            hull.pop()
        hull.append(np.asarray(pt))
    lower = hull
    hull = []
    for pt in p[::-1]:
        while len(hull) >= 2 and np.cross(hull[-1] - hull[-2], np.asarray(pt) - hull[-1]) <= 1e-12:
            hull.pop()
        hull.append(np.asarray(pt))
    return np.asarray(lower[:-1] + hull[:-1])


def epsilon_quality(pts: np.ndarray) -> float:
    hull = convex_hull_2d(pts)
    if len(hull) < 3:
        return 0.0
    if not _origin_inside(hull):
        return 0.0
    best = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ab = b - a
        d = abs(ab[0] * (-a[1]) - (-a[0]) * ab[1]) / (np.linalg.norm(ab) + 1e-12)
        best = min(best, float(d))
    return float(best)


def _origin_inside(hull: np.ndarray) -> bool:
    signs = []
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        c = (b[0] - a[0]) * (-a[1]) - (b[1] - a[1]) * (-a[0])
        signs.append(c > 0)
    s = set(signs)
    return len(s) == 1


def grasp_score(contacts: list[tuple[np.ndarray, np.ndarray]], mu: float, dim: int = 3) -> float:
    if not contacts:
        return 0.0
    w = np.vstack([contact_wrenches(p, n, mu) for p, n in contacts])
    hull = _convex_hull(w)
    if len(hull) < 3 or not _origin_inside(hull):
        return 0.0
    best = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ab = b - a
        d = abs(ab[0] * a[1] - ab[1] * a[0]) / (np.linalg.norm(ab) + 1e-12)
        best = min(best, float(d))
    _ = dim
    return float(best)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    hull: list[int] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = pts[i], pts[j]
            signs = set()
            for k in range(n):
                if k in (i, j):
                    continue
                c = (b[0] - a[0]) * (pts[k][1] - a[1]) - (b[1] - a[1]) * (pts[k][0] - a[0])
                if abs(c) > 1e-9:
                    signs.add(c > 0)
            if len(signs) <= 1:
                hull.extend([i, j])
    idx = sorted(set(hull))
    if len(idx) < 3:
        return pts[idx]
    ctr = pts[idx].mean(axis=0)
    order = sorted(idx, key=lambda i: np.arctan2(pts[i][1] - ctr[1], pts[i][0] - ctr[0]))
    return pts[order]


def bench_grasp_epsilon(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks: list[bool] = []
    lc = np.array([-1.0, 0.0])
    r = np.array([1.0, 0.0])
    nl = np.array([1.0, 0.0])
    nr = np.array([-1.0, 0.0])
    q_anti = grasp_score([(lc, nl), (r, nr)], mu=0.5)
    checks.append(q_anti > 0.05)
    q_low = grasp_score([(lc, nl), (r, nr)], mu=0.01)
    checks.append(q_anti > q_low > 0.0)
    q_single = grasp_score([(lc, nl)], mu=0.5)
    checks.append(q_single == 0.0)
    q_same = grasp_score([(lc, nl), (r, nl)], mu=0.5)
    checks.append(q_same == 0.0)
    top = np.array([0.0, 1.0])
    bot = np.array([0.0, -1.0])
    q4 = grasp_score([(lc, nl), (r, nr), (top, np.array([0, -1])), (bot, np.array([0, 1]))], 0.3)
    checks.append(q4 > 0.05)
    _ = rng
    score = float(np.mean(checks))
    return {"synthetic_grasp_epsilon": score}
