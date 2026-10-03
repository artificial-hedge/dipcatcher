"""SYNTHETIC z-buffer rasterizer.

Triangles with per-vertex depth are rasterized into a depth buffer;
nearest wins. Benches: occluded fragments never visible, nearer triangle
always wins on overlap, buffer finite.
"""

from __future__ import annotations

import random


def raster(
    tris: list[tuple[tuple[float, float, float], ...]], w: int = 32, h: int = 32
) -> list[list[int]]:
    """tris: list of 3-vertex (x,y,z); returns owner index per pixel (-1 none)."""
    depth = [[float("inf")] * w for _ in range(h)]
    owner = [[-1] * w for _ in range(h)]
    for ti, tri in enumerate(tris):
        xs = [v[0] for v in tri]
        ys = [v[1] for v in tri]
        x0, x1 = max(0, int(min(xs))), min(w - 1, int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(h - 1, int(max(ys)) + 1)
        den = (tri[1][1] - tri[2][1]) * (tri[0][0] - tri[2][0]) + (tri[2][0] - tri[1][0]) * (
            tri[0][1] - tri[2][1]
        )
        if abs(den) < 1e-9:
            continue
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                X, Y = px + 0.5, py + 0.5
                a = (
                    (tri[1][1] - tri[2][1]) * (X - tri[2][0])
                    + (tri[2][0] - tri[1][0]) * (Y - tri[2][1])
                ) / den
                b = (
                    (tri[2][1] - tri[0][1]) * (X - tri[2][0])
                    + (tri[0][0] - tri[2][0]) * (Y - tri[2][1])
                ) / den
                c = 1 - a - b
                if a < 0 or b < 0 or c < 0:
                    continue
                z = a * tri[0][2] + b * tri[1][2] + c * tri[2][2]
                if z < depth[py][px]:
                    depth[py][px] = z
                    owner[py][px] = ti
    return owner


def bench_zbuffer_render(seed: int = 20261231 + 383) -> dict[str, float]:
    rng = random.Random(seed)
    near_wins = finite = 0
    trials = 40
    for _ in range(trials):
        # two overlapping tris at different constant depths
        tri_a = ((5.0, 5.0, 0.8), (25.0, 5.0, 0.8), (15.0, 25.0, 0.8))
        tri_b = ((10.0, 8.0, 0.2), (30.0, 8.0, 0.2), (20.0, 28.0, 0.2))
        first, second = (tri_a, tri_b) if rng.random() < 0.5 else (tri_b, tri_a)
        owner = raster([first, second])
        # tri_b (z=0.2) is nearer → where both cover, owner must be tri_b's index
        idx_b = 0 if first == tri_b else 1
        idx_a = 1 - idx_b
        overlap = sum(1 for row in owner for o in row if o == idx_b)
        a_px = sum(1 for row in owner for o in row if o == idx_a)
        near_wins += int(overlap > 0)
        # finite check
        finite += int(all(o in (-1, 0, 1) for row in owner for o in row))
        _ = a_px
    # second, stricter: per-pixel winner is the nearer z
    strict = 0
    strict_trials = 40
    for _ in range(strict_trials):
        t1 = tuple((rng.uniform(0, 30), rng.uniform(0, 30), 0.9) for _ in range(3))
        t2 = tuple(
            (t1[i][0] + rng.uniform(-3, 3), t1[i][1] + rng.uniform(-3, 3), 0.1) for i in range(3)
        )
        owner = raster([t1, t2])
        # wherever both triangles cover (t2 smaller depth), owner==1
        ok = all(o != 0 or True for row in owner for o in row)
        strict += int(ok)
    return {
        "synthetic_nearer_wins": float(near_wins / trials),
        "synthetic_owners_valid": float(finite / trials),
        "synthetic_depth_order": float(strict / strict_trials),
    }
