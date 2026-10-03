"""Two-pass chamfer distance transform (approximate Euclidean) with medial
ridges (SYNTHETIC bench only)."""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 992

_W1, _W2 = 1.0, np.sqrt(2.0)


def chamfer_dt(mask: np.ndarray) -> np.ndarray:
    n, m = mask.shape
    inf = 1e9
    d = np.where(mask, 0.0, inf)
    for i in range(n):
        for j in range(m):
            if d[i, j] == 0:
                continue
            best = d[i, j]
            if i > 0:
                best = min(best, d[i - 1, j] + _W1)
            if j > 0:
                best = min(best, d[i, j - 1] + _W1)
            if i > 0 and j > 0:
                best = min(best, d[i - 1, j - 1] + _W2)
            if i > 0 and j + 1 < m:
                best = min(best, d[i - 1, j + 1] + _W2)
            d[i, j] = best
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = d[i, j]
            if i + 1 < n:
                best = min(best, d[i + 1, j] + _W1)
            if j + 1 < m:
                best = min(best, d[i, j + 1] + _W1)
            if i + 1 < n and j + 1 < m:
                best = min(best, d[i + 1, j + 1] + _W2)
            if i + 1 < n and j > 0:
                best = min(best, d[i + 1, j - 1] + _W2)
            d[i, j] = best
    return d


def medial_ridge(d: np.ndarray) -> np.ndarray:
    n, m = d.shape
    ridge = np.zeros((n, m), bool)
    for i in range(n):
        for j in range(m):
            if d[i, j] <= 0:
                continue
            nbrs = [
                d[i + di, j + dj]
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
                if (di or dj) and 0 <= i + di < n and 0 <= j + dj < m
            ]
            if nbrs and d[i, j] >= max(nbrs) - 1e-9:
                ridge[i, j] = True
    return ridge


def bench_distance_transform(seed: int = _SEED) -> dict[str, float]:
    mask = np.zeros((30, 30), bool)
    mask[10:20, 10:20] = True
    d = chamfer_dt(mask)
    checks = [abs(float(d[0, 0]) - np.hypot(10, 10)) < 0.5]
    checks.append(abs(float(d[29, 29]) - np.hypot(10, 10)) < 0.6)
    checks.append(float(d[10, 10]) == 0.0)
    checks.append(abs(float(d[0, 15]) - 10.0) < 0.01)
    line = np.zeros((30, 30), bool)
    line[15, 5:25] = True
    d2 = chamfer_dt(line)
    interior = chamfer_dt(~mask)
    ridge = medial_ridge(interior)
    checks.append(float(interior[15, 15]) == float(interior.max()))
    checks.append(int(ridge.sum()) > 0 and int((ridge & ~mask).sum()) == 0)
    checks.append(float(d2[0, 15]) - 15.0 == 0.0)
    return {"synthetic_distance_transform": float(np.mean(checks))}
