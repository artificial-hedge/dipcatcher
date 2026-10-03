"""Ear-clipping triangulation of simple polygons — SYNTHETIC benches.

`triangulate(poly)` returns index triples; verified by exact area
conservation and diagonal-inside tests.
"""

from __future__ import annotations

import math
import random

Pt = tuple[float, float]
Poly = list[Pt]


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _area(poly: Poly) -> float:
    return (
        abs(
            sum(
                poly[i][0] * poly[(i + 1) % len(poly)][1]
                - poly[(i + 1) % len(poly)][0] * poly[i][1]
                for i in range(len(poly))
            )
        )
        / 2
    )


def _inside(p: Pt, a: Pt, b: Pt, c: Pt) -> bool:
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def triangulate(poly: Poly) -> list[tuple[int, int, int]]:
    """Ear-clipping O(n^2); returns (i,j,k) index triples."""
    n = len(poly)
    if n < 3:
        return []
    # orientation: ensure CCW
    if _area_signed(poly) < 0:
        idx = list(reversed(range(n)))
    else:
        idx = list(range(n))
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        clipped = False
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k % m], idx[(k + 1) % m], idx[(k + 2) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if _cross(a, b, c) <= 1e-12:
                continue  # reflex or degenerate
            if any(_inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            del idx[(k + 1) % m]
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def _area_signed(poly: Poly) -> float:
    return (
        sum(
            poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
            for i in range(len(poly))
        )
        / 2
    )


def _star_poly(rng: random.Random, n: int) -> Poly:
    # angle-sorted star polygon — simple
    pts: Poly = []
    for k in range(n):
        ang = 2 * 3.141592653589793 * k / n
        r = 0.4 + rng.random()
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    return pts


def bench_ear_clipping(seed: int = 20261231 + 320) -> dict[str, float]:
    rng = random.Random(seed)
    area_ok = count_ok = 0
    trials = 30
    for _ in range(trials):
        n = rng.randrange(5, 25)
        poly = _star_poly(rng, n)
        tris = triangulate(poly)
        count_ok += int(len(tris) == n - 2)
        s = 0.0
        for i, j, k in tris:
            s += abs(_area_signed([poly[i], poly[j], poly[k]]))
        area_ok += int(abs(s - _area(poly)) < 1e-6 * max(1.0, _area(poly)))
    return {
        "synthetic_area_conserved": float(area_ok / trials),
        "synthetic_triangle_count": float(count_ok / trials),
    }
