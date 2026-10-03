"""Fast-marching eikonal traveltime solver (Godunov upwind, heap march).

Solves |grad T| = s(x) (slowness) on a 2-D grid via the
Sethian fast marching method with the standard quadratic update.
"""

from __future__ import annotations

import heapq

import numpy as np

_SEED = 20261231 + 895


def _upd(t1: float, t2: float, s: float, h: float) -> float:
    """Godunov update from two perpendicular neighbors (inf if absent)."""
    a, b = min(t1, t2), max(t1, t2)
    if b - a >= s * h:
        return a + s * h
    disc = 2.0 * s * s * h * h - (b - a) ** 2
    if disc < 0:
        return a + s * h
    return float(0.5 * (a + b + np.sqrt(disc)))


def fast_march(slow: np.ndarray, src: tuple[int, int], h: float = 1.0) -> np.ndarray:
    """Traveltimes on grid `slow` (s/m or s/cell) from source index."""
    s = np.asarray(slow, dtype=np.float64)
    nz, nx = s.shape
    inf = np.inf
    T = np.full((nz, nx), inf)
    T[src] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, src[0], src[1])]
    done = np.zeros((nz, nx), dtype=bool)
    while heap:
        tt, z, x = heapq.heappop(heap)
        if done[z, x]:
            continue
        done[z, x] = True
        for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            z2, x2 = z + dz, x + dx
            if not (0 <= z2 < nz and 0 <= x2 < nx) or done[z2, x2]:
                continue
            tv = T[z2, max(0, x2 - 1) : min(nx, x2 + 2)]
            th = T[max(0, z2 - 1) : min(nz, z2 + 2), x2]
            t1 = float(np.min(tv))
            t2 = float(np.min(th))
            cand = _upd(t1, t2, s[z2, x2], h)
            if cand < T[z2, x2]:
                T[z2, x2] = cand
                heapq.heappush(heap, (cand, z2, x2))
    return T


def head_wave_time(x: float, v1: float, v2: float, z: float, zs: float = 0.0) -> float:
    """1-D analytic refracted (head-wave) traveltime for a flat interface at depth z.

    Source/receiver at surface (depth zs ~ 0): T = 2*z*cos(ic)/v1 + x/v2
    where ic = critical angle asin(v1/v2).
    """
    ic = np.arcsin(v1 / v2)
    return float(2.0 * (z - zs) * np.cos(ic) / v1 + x / v2)


def bench_eikonal_fmm(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    _ = rng
    h = 5.0
    nz, nx = 80, 120
    score = 0.0
    # constant-velocity: T = dist/v
    v = 2000.0
    slow = np.full((nz, nx), 1.0 / v)
    src = (10, 20)
    T = fast_march(slow, src, h)
    zz, xx = np.mgrid[0:nz, 0:nx]
    d = np.sqrt(((zz - src[0]) * h) ** 2 + ((xx - src[1]) * h) ** 2)
    err = float(np.max(np.abs(T - d / v)[d > 3 * h]) / np.max(d / v))
    score += 1.0 if err < 0.03 else 0.0
    # two-layer: head wave beats direct beyond crossover distance
    v1, v2, zint = 1500.0, 4000.0, 200.0
    nx2 = 160
    slow2 = np.full((nz, nx2), 1.0 / v1)
    slow2[int(zint / h) :, :] = 1.0 / v2
    src2 = (2, 5)
    T2 = fast_march(slow2, src2, h)
    xf = 700.0  # crossover for these params is ~594 m
    ix = int(xf / h) + 5
    t_model = T2[2, ix]
    t_direct = xf / v1
    t_head = head_wave_time(xf, v1, v2, zint)
    # model should pick the faster of refracted/direct (head wave here)
    score += 1.0 if t_model < t_direct else 0.0
    score += 1.0 if abs(t_model - t_head) / t_head < 0.08 else 0.0
    # causality: traveltime nondecreasing along any ray direction
    row = T[10, :]
    score += 1.0 if bool(np.all(np.diff(row[src[1] :]) >= -1e-9)) else 0.0
    return {"synthetic_eikonal_fmm": score / 4.0}
