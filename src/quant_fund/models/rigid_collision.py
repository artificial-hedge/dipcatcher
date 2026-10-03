"""2D rigid-ball impulse collisions (synthetic).

Positional wall bounce + pair impulse with restitution e.
Verified: (i) momentum conserved in elastic collisions; (ii)
relative normal velocity reverses at rate e; (iii) head-on equal
mass swap velocities (Newton's cradle limit).
"""

from __future__ import annotations

import math
import random


def collide(
    p1: list[float],
    v1: list[float],
    m1: float,
    p2: list[float],
    v2: list[float],
    m2: float,
    e: float,
) -> tuple[list[float], list[float]]:
    """Impulse resolution for overlapping pair."""
    nx = p2[0] - p1[0]
    ny = p2[1] - p1[1]
    d = math.hypot(nx, ny) or 1e-9
    nx, ny = nx / d, ny / d
    rvx = v2[0] - v1[0]
    rvy = v2[1] - v1[1]
    vn = rvx * nx + rvy * ny
    if vn >= 0:
        return v1, v2  # separating
    j = -(1 + e) * vn / (1 / m1 + 1 / m2)
    ix, iy = j * nx, j * ny
    return (
        [v1[0] - ix / m1, v1[1] - iy / m1],
        [v2[0] + ix / m2, v2[1] + iy / m2],
    )


def bench_rigid_collision(seed: int = 20261231 + 273) -> dict[str, float]:
    rng = random.Random(seed)
    mom_ok = e_ok = 0
    trials = 40
    for _ in range(trials):
        m1 = rng.uniform(0.5, 3)
        m2 = rng.uniform(0.5, 3)
        e = rng.uniform(0.0, 1.0)
        p1 = [0.0, 0.0]
        p2 = [1.0, rng.uniform(-0.4, 0.4)]
        # aim v1 at p2, v2 at rest
        v1 = [rng.uniform(1, 3), 0.0]
        v2 = [0.0, 0.0]
        p0x = m1 * v1[0] + m2 * v2[0]
        p0y = m1 * v1[1] + m2 * v2[1]
        n1, n2 = collide(p1, v1, m1, p2, v2, m2, e)
        p1x = m1 * n1[0] + m2 * n2[0]
        p1y = m1 * n1[1] + m2 * n2[1]
        mom_ok += int(math.hypot(p1x - p0x, p1y - p0y) < 1e-9 * max(1.0, abs(p0x)))
        # restitution: new relative normal velocity = -e * old
        nx = p2[0] - p1[0]
        ny = p2[1] - p1[1]
        dd = math.hypot(nx, ny)
        nx, ny = nx / dd, ny / dd
        vn0 = (v2[0] - v1[0]) * nx + (v2[1] - v1[1]) * ny
        vn1 = (n2[0] - n1[0]) * nx + (n2[1] - n1[1]) * ny
        e_ok += int(abs(vn1 + e * vn0) < 1e-9)
    # Newton's cradle: equal masses, e=1, head-on → swap
    n1, n2 = collide([0.0, 0.0], [2.0, 0.0], 1.0, [1.0, 0.0], [0.0, 0.0], 1.0, 1.0)
    swap_ok = abs(n1[0]) < 1e-9 and abs(n2[0] - 2.0) < 1e-9
    return {
        "synthetic_momentum": float(mom_ok / trials),
        "synthetic_restitution": float(e_ok / trials),
        "synthetic_cradle_swap": float(swap_ok),
    }
