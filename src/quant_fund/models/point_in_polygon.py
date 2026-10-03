"""Point-in-polygon via ray casting — SYNTHETIC benches.

Verified against a winding-number oracle and Monte-Carlo area estimate
on a known-shape polygon.
"""

from __future__ import annotations

import math
import random

Pt = tuple[float, float]


def inside_ray(p: Pt, poly: list[Pt]) -> bool:
    n = len(poly)
    c = False
    j = n - 1
    for i in range(n):
        a, b = poly[i], poly[j]
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            c = not c
        j = i
    return c


def inside_winding(p: Pt, poly: list[Pt]) -> bool:
    n = len(poly)
    w = 0
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if a[1] <= p[1]:
            if b[1] > p[1] and _cross(a, b, p) > 0:
                w += 1
        elif b[1] <= p[1] and _cross(a, b, p) < 0:
            w -= 1
    return w != 0


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def bench_point_in_polygon(seed: int = 20261231 + 323) -> dict[str, float]:
    rng = random.Random(seed)
    agree = 0
    trials = 30
    total_pts = 0
    for _ in range(trials):
        n = rng.randrange(5, 20)
        poly = []
        for k in range(n):
            ang = 2 * math.pi * k / n
            r = 0.5 + rng.random()
            poly.append((r * math.cos(ang), r * math.sin(ang)))
        for _ in range(40):
            p = (rng.uniform(-2, 2), rng.uniform(-2, 2))
            total_pts += 1
            agree += int(inside_ray(p, poly) == inside_winding(p, poly))
    # Monte-Carlo area estimate on unit square vs truth
    sq = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    inside = sum(inside_ray((rng.uniform(-2, 2), rng.uniform(-2, 2)), sq) for _ in range(4000))
    est = 16.0 * inside / 4000
    return {
        "synthetic_ray_vs_winding": float(agree / total_pts),
        "synthetic_mc_area_err": float(abs(est - 4.0)),
        "synthetic_mc_area_ok": float(abs(est - 4.0) < 0.4),
    }
