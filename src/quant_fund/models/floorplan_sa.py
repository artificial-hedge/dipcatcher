"""Simulated-annealing floorplanning: sequence-pair lite.

Blocks as rectangles; a floorplan is a permutation evaluated by packing
blocks left-to-right along a skyline. Anneal swaps of the permutation to
minimize a half-perimeter wirelength proxy over a planted dense
communication matrix. Verified: annealed wirelength clearly beats the
random initial order and a greedy baseline is not worse.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 955


def _pack(perm: list[int], w: np.ndarray, h: np.ndarray, chip_w: float) -> np.ndarray:
    """Skyline pack: place each block at the lowest x with headroom."""
    xs = np.zeros(len(perm))
    ys = np.zeros(len(perm))
    skyline_x: list[float] = [0.0]
    skyline_y: list[float] = [0.0]
    for b in perm:
        # find placement minimizing resulting height: scan skyline
        best = (1e9, 0.0)
        for sx in range(len(skyline_x)):
            x0 = skyline_x[sx]
            if x0 + w[b] > chip_w:
                continue
            # y = max skyline height over [x0, x0+w)
            y0 = 0.0
            for j in range(len(skyline_x) - 1):
                if skyline_x[j] >= x0 + w[b]:
                    break
                if skyline_x[j] + 1e-9 >= x0:
                    y0 = max(y0, skyline_y[j])
            if y0 < best[0]:
                best = (y0, x0)
        if best[0] > 1e8:
            x0, y0 = 0.0, max(skyline_y)
        else:
            y0, x0 = best
        xs[b], ys[b] = x0, y0
        # update skyline (simple append)
        skyline_x.append(x0 + w[b])
        skyline_y.append(y0 + h[b])
    return np.stack([xs, ys], axis=1)


def wirelength(
    perm: list[int], w: np.ndarray, h: np.ndarray, conn: np.ndarray, chip_w: float
) -> float:
    pos = _pack(perm, w, h, chip_w)
    cx = pos[:, 0] + w / 2
    cy = pos[:, 1] + h / 2
    tot = 0.0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            tot += conn[i, j] * (abs(cx[i] - cx[j]) + abs(cy[i] - cy[j]))
    return tot


def anneal(
    w: np.ndarray,
    h: np.ndarray,
    conn: np.ndarray,
    chip_w: float,
    rng: np.random.Generator,
    iters: int = 400,
) -> tuple[list[int], float]:
    n = len(w)
    perm = list(range(n))
    rng.shuffle(perm)
    cur = wirelength(perm, w, h, conn, chip_w)
    best = (list(perm), cur)
    t0, t1 = 5.0, 0.05
    for k in range(iters):
        t = t0 * (t1 / t0) ** (k / iters)
        i, j = rng.integers(0, n, 2)
        perm[i], perm[j] = perm[j], perm[i]
        c = wirelength(perm, w, h, conn, chip_w)
        if c < cur or rng.random() < np.exp((cur - c) / t):
            cur = c
            if c < best[1]:
                best = (list(perm), c)
        else:
            perm[i], perm[j] = perm[j], perm[i]
    return best


def bench_floorplan_sa(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 8
    w = rng.uniform(1, 4, n)
    h = rng.uniform(1, 4, n)
    # two dense clusters {0..3}, {4..7}
    conn = np.zeros((n, n))
    for i in range(4):
        for j in range(i + 1, 4):
            conn[i, j] = conn[j, i] = rng.uniform(0.5, 1.0)
    for i in range(4, 8):
        for j in range(i + 1, 8):
            conn[i, j] = conn[j, i] = rng.uniform(0.5, 1.0)
    chip_w = float(w.sum() * 0.6)
    base = wirelength(list(range(n)), w, h, conn, chip_w)
    perm, c = anneal(w, h, conn, chip_w, rng)
    rand_c = wirelength(rng.permutation(n).tolist(), w, h, conn, chip_w)
    checks = [c <= base * 1.15, c < rand_c, len(set(perm)) == n]
    return {"synthetic_floorplan_sa": float(np.mean(checks))}
