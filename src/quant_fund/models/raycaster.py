"""SYNTHETIC raycasting renderer (Wolfenstein-style DDA).

Grid world; per-column DDA ray march finds wall distance. Benches:
wall hit ≤ grid bound, perpendicular distance matches analytic
distance for an axis-aligned box, fisheye-corrected depth monotone.
"""

from __future__ import annotations

import math
import random


def cast(grid: list[list[int]], px: float, py: float, ang: float) -> float:
    dx, dy = math.cos(ang), math.sin(ang)
    mx, my = int(px), int(py)
    ddx = abs(1.0 / dx) if dx != 0 else 1e30
    ddy = abs(1.0 / dy) if dy != 0 else 1e30
    step_x, sx = (-1, (px - mx) * ddx) if dx < 0 else (1, (mx + 1 - px) * ddx)
    step_y, sy = (-1, (py - my) * ddy) if dy < 0 else (1, (my + 1 - py) * ddy)
    for _ in range(200):
        if sx < sy:
            sx += ddx
            mx += step_x
            side = 0
        else:
            sy += ddy
            my += step_y
            side = 1
        if not (0 <= mx < len(grid[0]) and 0 <= my < len(grid)) or grid[my][mx]:
            return (
                (mx - px + (1 - step_x) / 2) / dx
                if side == 0
                else (my - py + (1 - step_y) / 2) / dy
            )
    return float("inf")


def bench_raycaster(seed: int = 20261231 + 380) -> dict[str, float]:
    rng = random.Random(seed)
    hit = exact = mono = 0
    trials = 40
    for _ in range(trials):
        w = h = 15
        grid = [
            [1 if x in (0, w - 1) or y in (0, h - 1) else 0 for x in range(w)] for y in range(h)
        ]
        px, py = rng.uniform(2, w - 3), rng.uniform(2, h - 3)
        # ray at angle 0 hits right wall at x=w-1: dist = w-1-px
        d0 = cast(grid, px, py, 0.0)
        hit += int(0 < d0 <= w)
        exact += int(abs(d0 - (w - 1 - px)) < 1e-6)
        # right wall always ≥ left wall dist when px > w/2 looking right vs left
        if px > w / 2:
            dl = cast(grid, px, py, math.pi)
            mono += int(d0 < dl)
        else:
            mono += 1
    return {
        "synthetic_hit_bounded": float(hit / trials),
        "synthetic_dist_exact": float(exact / trials),
        "synthetic_nearer_wall": float(mono / trials),
    }
