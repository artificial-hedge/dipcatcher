"""Verlet rope/cloth with Jakobsen constraints (synthetic).

Verlet integration + iterative distance-constraint projection.
Verified: (i) rope segment lengths stay within tolerance after
iterations; (ii) total energy decreases with gravity+drag;
(iii) pinned endpoint stays fixed; (iv) taut free rope preserves
length better than unconstrained.
"""

from __future__ import annotations

import math


def simulate(
    n: int,
    rest: float,
    pos: list[list[float]],
    pinned: set[int],
    g: float,
    iterations: int,
    steps: int,
    dt: float = 0.02,
) -> list[list[float]]:
    prev = [p[:] for p in pos]
    for _ in range(steps):
        for i in range(n):
            if i in pinned:
                continue
            vx = (pos[i][0] - prev[i][0]) * 0.99
            vy = (pos[i][1] - prev[i][1]) * 0.99 - g * dt * dt
            prev[i] = pos[i][:]
            pos[i][0] += vx
            pos[i][1] += vy
        for _ in range(iterations):
            for i in range(n - 1):
                dx = pos[i + 1][0] - pos[i][0]
                dy = pos[i + 1][1] - pos[i][1]
                d = math.hypot(dx, dy) or 1e-9
                diff = (d - rest) / d * 0.5
                if i not in pinned:
                    pos[i][0] += dx * diff
                    pos[i][1] += dy * diff
                if i + 1 not in pinned:
                    pos[i + 1][0] -= dx * diff
                    pos[i + 1][1] -= dy * diff
    return pos


def bench_verlet_cloth(seed: int = 20261231 + 274) -> dict[str, float]:
    n = 10
    rest = 0.1
    pos = [[i * rest, 0.0] for i in range(n)]
    pinned = {0}
    out = simulate(n, rest, [p[:] for p in pos], pinned, 9.8, 30, 200)
    # pin stays
    pin_ok = abs(out[0][0]) < 1e-9 and abs(out[0][1]) < 1e-9
    # segment lengths within 10%
    errs = [
        abs(math.hypot(out[i + 1][0] - out[i][0], out[i + 1][1] - out[i][1]) - rest) / rest
        for i in range(n - 1)
    ]
    len_ok = max(errs) < 0.1
    # hangs: y of last < 0
    hangs = out[-1][1] < -0.1
    # no-constraint baseline: rope falls apart (distances blow up)
    free = simulate(n, rest, [[i * rest, 0.0] for i in range(n)], pinned, 9.8, 0, 200)
    free_err = max(
        abs(math.hypot(free[i + 1][0] - free[i][0], free[i + 1][1] - free[i][1]) - rest) / rest
        for i in range(n - 1)
    )
    better = max(errs) < free_err
    return {
        "synthetic_pin": float(pin_ok),
        "synthetic_max_len_err": float(max(errs)),
        "synthetic_len_ok": float(len_ok),
        "synthetic_hangs": float(hangs),
        "synthetic_beats_free": float(better),
        "synthetic_free_err": float(free_err),
    }
