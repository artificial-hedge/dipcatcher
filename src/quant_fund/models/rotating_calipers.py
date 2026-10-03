"""Rotating calipers — convex-hull diameter vs brute-force oracle."""

from __future__ import annotations

import random

Pt = tuple[float, float]


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(pts: list[Pt]) -> list[Pt]:
    p = sorted(set(pts))
    if len(p) <= 1:
        return p
    lo: list[Pt] = []
    for x in p:
        while len(lo) >= 2 and _cross(lo[-2], lo[-1], x) <= 0:
            lo.pop()
        lo.append(x)
    hi: list[Pt] = []
    for x in reversed(p):
        while len(hi) >= 2 and _cross(hi[-2], hi[-1], x) <= 0:
            hi.pop()
        hi.append(x)
    return lo[:-1] + hi[:-1]


def _d2(a: Pt, b: Pt) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def diameter(h: list[Pt]) -> float:
    """Rotating-calipers max distance on convex hull (returns distance)."""
    n = len(h)
    if n < 2:
        return 0.0
    j = 1
    best = 0.0
    for i in range(n):
        ni = (i + 1) % n
        while _d2(h[i], h[(j + 1) % n]) > _d2(h[i], h[j]):
            j = (j + 1) % n
        best = max(best, _d2(h[i], h[j]), _d2(h[ni], h[j]))
    return float(best**0.5)


def bench_rotating_calipers(seed: int = 20261231 + 325) -> dict[str, float]:
    rng = random.Random(seed)
    ok = hull_ok = 0
    trials = 40
    for _ in range(trials):
        pts = [(rng.uniform(-30, 30), rng.uniform(-30, 30)) for _ in range(rng.randrange(10, 120))]
        h = hull(pts)
        dia = diameter(h)
        brute = max((_d2(a, b) for a in h for b in h), default=0.0) ** 0.5
        ok += int(abs(dia - brute) < 1e-9)
        # hull vertices form convex polygon: all cross signs equal
        m = len(h)
        signs = {_cross(h[i], h[(i + 1) % m], h[(i + 2) % m]) > 0 for i in range(m)}
        hull_ok += int(len(signs) == 1 or m < 3)
    return {
        "synthetic_diameter_ok": float(ok / trials),
        "synthetic_hull_convex": float(hull_ok / trials),
    }
