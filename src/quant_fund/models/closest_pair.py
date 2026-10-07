"""Closest pair of points — divide & conquer vs O(n^2) oracle (SYNTHETIC)."""

from __future__ import annotations

import random

Pt = tuple[float, float]


def _d(a: Pt, b: Pt) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def closest(pts: list[Pt]) -> float:
    px = sorted(pts)
    return _rec(px)


def _rec(px: list[Pt]) -> float:
    n = len(px)
    if n <= 3:
        return min(
            (_d(px[i], px[j]) for i in range(n) for j in range(i + 1, n)), default=float("inf")
        )
    mid = n // 2
    xl = px[mid][0]
    dl = _rec(px[:mid])
    dr = _rec(px[mid:])
    d = min(dl, dr)
    strip = [p for p in px if abs(p[0] - xl) < d]
    strip.sort(key=lambda p: p[1])
    for i in range(len(strip)):
        for j in range(i + 1, min(i + 7, len(strip))):
            d = min(d, _d(strip[i], strip[j]))
    return d


def _brute(pts: list[Pt]) -> float:
    return min(
        (_d(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))),
        default=float("inf"),
    )


def bench_closest_pair(seed: int = 20261231 + 324) -> dict[str, float]:
    rng = random.Random(seed)
    ok = 0
    trials = 40
    for _ in range(trials):
        pts = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(rng.randrange(10, 200))]
        ok += int(abs(closest(pts) - _brute(pts)) < 1e-9)
    return {"synthetic_matches_brute": float(ok / trials)}
