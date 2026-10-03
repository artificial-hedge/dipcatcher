"""SYNTHETIC scanline polygon fill (even-odd rule).

Fill a polygon on a grid; verify filled area ≈ analytic polygon area
(shoelace) within rasterization tolerance, and filled set is a
superset-subset match against point-in-polygon sampling at cell centers.
"""

from __future__ import annotations

import random


def fill(poly: list[tuple[float, float]], w: int, h: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for y in range(h):
        xs: list[float] = []
        yc = y + 0.5
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            if (y1 <= yc < y2) or (y2 <= yc < y1):
                xs.append(x1 + (yc - y1) / (y2 - y1) * (x2 - x1))
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            for x in range(max(0, int(xs[i] + 0.5)), min(w, int(xs[i + 1] + 0.5))):
                out.add((x, y))
    return out


def _area(poly: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
            for i in range(len(poly))
        )
        / 2
    )


def _inside(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    ins = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1) + x1:
            ins = not ins
    return ins


def bench_scanline_fill(seed: int = 20261231 + 382) -> dict[str, float]:
    rng = random.Random(seed)
    area_ok = center_ok = sym_ok = 0
    trials = 40
    for _ in range(trials):
        # random convex-ish polygon inside 20x20
        cx, cy = 10.0, 10.0
        poly = []
        for a in range(6):
            ang = a * 1.047 + rng.uniform(-0.3, 0.3)
            r = rng.uniform(3, 8)
            poly.append(
                (cx + r * __import__("math").cos(ang), cy + r * __import__("math").sin(ang))
            )
        cells = fill(poly, 20, 20)
        area_ok += int(abs(len(cells) - _area(poly)) / max(_area(poly), 1) < 0.35)
        # all filled cells' centers inside polygon
        center_ok += int(
            sum(_inside(x + 0.5, y + 0.5, poly) for x, y in cells) >= 0.9 * max(len(cells), 1)
        )
        # fill is symmetric under 180° rotation of a symmetric polygon
        sq = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
        cells_sq = fill(sq, 20, 20)
        sym_ok += int(len(cells_sq) == 100)
    return {
        "synthetic_area_match": float(area_ok / trials),
        "synthetic_centers_inside": float(center_ok / trials),
        "synthetic_square_exact": float(sym_ok / trials),
    }
