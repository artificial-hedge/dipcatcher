"""SYNTHETIC Bresenham line rasterizer.

Integer-only DDA; verifies: every rasterized point lies within half a
pixel of the true segment, reversal symmetry, and 8-connected steps.
"""

from __future__ import annotations

import random


def bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return pts


def _dist_seg(p: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> float:
    dx, dy = x1 - x0, y1 - y0
    t = max(0.0, min(1.0, ((p[0] - x0) * dx + (p[1] - y0) * dy) / (dx * dx + dy * dy)))
    qx, qy = x0 + t * dx, y0 + t * dy
    return float(((p[0] - qx) ** 2 + (p[1] - qy) ** 2) ** 0.5)


def bench_bresenham_line(seed: int = 20261231 + 381) -> dict[str, float]:
    rng = random.Random(seed)
    exact = sym = conn = 0
    n = 0
    trials = 60
    for _ in range(trials):
        x0, y0 = rng.randrange(-20, 21), rng.randrange(-20, 21)
        x1, y1 = rng.randrange(-20, 21), rng.randrange(-20, 21)
        if (x0, y0) == (x1, y1):
            continue
        n += 1
        pts = bresenham(x0, y0, x1, y1)
        exact += int(all(_dist_seg(p, x0, y0, x1, y1) < 0.51 for p in pts))
        rev = bresenham(x1, y1, x0, y0)
        # Bresenham is not symmetric; forward/reverse differ by ≤1 px
        hd = max(min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in rev) for p in pts)
        hr = max(min(max(abs(q[0] - p[0]), abs(q[1] - p[1])) for p in pts) for q in rev)
        sym += int(max(hd, hr) <= 1)
        conn += int(
            all(
                abs(pts[i + 1][0] - pts[i][0]) <= 1 and abs(pts[i + 1][1] - pts[i][1]) <= 1
                for i in range(len(pts) - 1)
            )
        )
    return {
        "synthetic_within_half_px": float(exact / max(n, 1)),
        "synthetic_reverse_hausdorff_le1": float(sym / max(n, 1)),
        "synthetic_8_connected": float(conn / max(n, 1)),
    }
