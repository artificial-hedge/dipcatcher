"""Marker-controlled watershed segmentation by sorted flooding
(Vincent-Soille) (SYNTHETIC bench only)."""

from __future__ import annotations

import heapq

import numpy as np

_SEED = 20261231 + 989


def grad_magnitude(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img)
    return np.asarray(np.hypot(gx, gy), dtype=np.float64)


def local_minima_markers(img: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    n, m = img.shape
    markers = np.zeros((n, m), int)
    label = 0
    for i in range(n):
        for j in range(m):
            if img[i, j] > thresh:
                continue
            nbr = [
                markers[i + di, j + dj]
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
                if 0 <= i + di < n and 0 <= j + dj < m and markers[i + di, j + dj] > 0
            ]
            if nbr:
                markers[i, j] = min(nbr)
            else:
                label += 1
                markers[i, j] = label
    return markers


def watershed(cost: np.ndarray, markers: np.ndarray) -> np.ndarray:
    n, m = cost.shape
    lab = np.where(markers > 0, markers, 0)
    heap: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(m):
            if lab[i, j] > 0:
                heapq.heappush(heap, (float(cost[i, j]), i, j))
    in_queue = lab > 0
    while heap:
        c, i, j = heapq.heappop(heap)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m and lab[ni, nj] == 0 and not in_queue[ni, nj]:
                    lab[ni, nj] = lab[i, j]
                    in_queue[ni, nj] = True
                    heapq.heappush(heap, (float(cost[ni, nj]), ni, nj))
    return lab


def bench_watershed_seg(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 50
    yy, xx = np.mgrid[0:n, 0:n]
    d1 = np.hypot(yy - 20, xx - 18)
    d2 = np.hypot(yy - 20, xx - 32)
    img = np.minimum(1.0 - np.clip(10 - d1, 0, 10) / 10, 1.0 - np.clip(10 - d2, 0, 10) / 10)
    img = np.clip(img + 0.02 * rng.standard_normal((n, n)), 0, 1)
    markers = local_minima_markers(img, 0.15)
    checks = [int(markers.max()) == 2]
    lab = watershed(img, markers)
    l1 = int(markers[20, 18])
    l2 = int(markers[20, 32])
    region1 = lab == l1
    region2 = lab == l2
    checks.append(region1[20, 18] and not region1[20, 32])
    checks.append(region2[20, 32] and not region2[20, 18])
    boundary = lab[:, 24:27]
    checks.append(int((boundary == l1).sum()) + int((boundary == l2).sum()) >= 40)
    checks.append(int((lab > 0).sum()) == n * n)
    return {"synthetic_watershed_seg": float(np.mean(checks))}
