"""Fast-marching canon: Sethian FMM for the eikonal |∇T| = 1 —
narrow-band Dijkstra-style propagation with the quadratic
upwind update (Godunov). Exact check against the Euclidean
distance from a point source; wall/obstacle variant included.
All SYNTHETIC.
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _quad_solve(t_a: float, t_b: float, dx: float) -> float:
    """Upwind Godunov update for |∇T|=1 on a 2-D grid.

    (T − t_a)²₊ + (T − t_b)²₊ = dx² — solve the larger root,
    fall back to the 1-D update when the discriminant fails.
    """
    ta, tb = sorted((t_a, t_b))
    disc = 2 * dx**2 - (tb - ta) ** 2
    if disc >= 0 and (tb - ta) / dx < np.sqrt(2):
        t = (ta + tb + np.sqrt(disc)) / 2
        if t > tb:
            return float(t)
    return ta + dx


def fast_marching(
    nx: int,
    ny: int,
    seeds: list[tuple[int, int, float]],
    dx: float = 1.0,
    blocked: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """FMM travel time on an nx×ny grid.

    seeds: (i, j, t0) source cells. blocked: boolean mask of
    impassable cells (T stays inf). Returns the travel-time field.
    """
    T = np.full((nx, ny), np.inf)
    state = np.zeros((nx, ny), dtype=np.int8)  # 0 far, 1 narrow, 2 known
    heap: list[tuple[float, int, int]] = []
    for i, j, t0 in seeds:
        T[i, j] = t0
        heapq.heappush(heap, (t0, i, j))
        state[i, j] = 1
    while heap:
        t, i, j = heapq.heappop(heap)
        if state[i, j] == 2:
            continue
        state[i, j] = 2
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if blocked is not None and blocked[ni, nj]:
                continue
            if state[ni, nj] == 2:
                continue
            # upwind neighbors along each axis
            txs = []
            for ii in (ni - 1, ni + 1):
                if 0 <= ii < nx and state[ii, nj] == 2:
                    txs.append(T[ii, nj])
            tys = []
            for jj in (nj - 1, nj + 1):
                if 0 <= jj < ny and state[ni, jj] == 2:
                    tys.append(T[ni, jj])
            ta = min(txs) if txs else np.inf
            tb = min(tys) if tys else np.inf
            if np.isinf(ta) and np.isinf(tb):
                continue
            if np.isinf(ta):
                t_new = tb + dx
            elif np.isinf(tb):
                t_new = ta + dx
            else:
                t_new = _quad_solve(ta, tb, dx)
            if t_new < T[ni, nj]:
                T[ni, nj] = t_new
                heapq.heappush(heap, (t_new, ni, nj))
                state[ni, nj] = 1
    return np.asarray(T, dtype=np.float64)


def bench_fast_marching(seed: int = 20261231) -> dict[str, float]:
    """FMM vs Euclidean distance from a center source + wall case."""
    out: dict[str, float] = {}
    n = 81
    c = n // 2
    dx = 1.0
    T = fast_marching(n, n, [(c, c, 0.0)], dx)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d = np.sqrt((ii - c) ** 2 + (jj - c) ** 2) * dx
    mask = np.isfinite(T) & (d > 1.0)
    rel = np.abs(T[mask] - d[mask]) / d[mask]
    out["synthetic_fmm_mean_rel_err"] = float(rel.mean())
    out["synthetic_fmm_max_rel_err"] = float(rel.max())
    # diagonal corner is the worst direction for grid FMM
    out["synthetic_fmm_corner_rel_err"] = float(abs(T[0, 0] - d[0, 0]) / d[0, 0])
    # wall with an off-axis gap: the path must detour through it
    blocked = np.zeros((n, n), dtype=np.bool_)
    blocked[c - 15 : c + 15, c - 5] = True
    blocked[c + 12, c - 5] = False  # gap above the target row
    Tw = fast_marching(n, n, [(c, c + 8, 0.0)], dx, blocked=blocked)
    left = Tw[c, c - 10]
    straight = abs((c - 10) - (c + 8)) * dx
    out["synthetic_fmm_wall_left"] = float(left)
    out["synthetic_fmm_wall_straight"] = float(straight)
    out["synthetic_fmm_detour"] = float(left / max(straight, 1e-30))
    out["synthetic_fmm_wall_finite"] = float(np.isfinite(left))
    return out
