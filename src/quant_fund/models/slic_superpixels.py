"""SLIC superpixels: k-means over (x, y, intensity) with spatially bounded
assignment (SYNTHETIC bench only)."""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 990


def slic(img: np.ndarray, k: int = 25, iters: int = 10, m: float = 10.0) -> np.ndarray:
    n, w = img.shape
    step = max(1, int(np.sqrt(n * w / k)))
    centers = []
    for i in range(step // 2, n, step):
        for j in range(step // 2, w, step):
            centers.append([float(i), float(j), float(img[i, j])])
    c = np.asarray(centers, dtype=np.float64)
    labels = -np.ones((n, w), int)
    dist = np.full((n, w), np.inf)
    for _ in range(iters):
        dist.fill(np.inf)
        for ci in range(len(c)):
            cy, cx, cl = c[ci]
            y0, y1 = int(max(0, cy - step)), int(min(n, cy + step + 1))
            x0, x1 = int(max(0, cx - step)), int(min(w, cx + step + 1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            ds = np.hypot(yy - cy, xx - cx) / step
            dc = np.abs(img[y0:y1, x0:x1] - cl) * m
            d = np.hypot(dc, ds)
            upd = d < dist[y0:y1, x0:x1]
            dist[y0:y1, x0:x1][upd] = d[upd]
            labels[y0:y1, x0:x1][upd] = ci
        for ci in range(len(c)):
            pts = np.argwhere(labels == ci)
            if len(pts) == 0:
                continue
            c[ci, 0] = pts[:, 0].mean()
            c[ci, 1] = pts[:, 1].mean()
            c[ci, 2] = float(img[pts[:, 0], pts[:, 1]].mean())
    return labels


def bench_slic_superpixels(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 40
    img = np.zeros((n, n))
    img[:, :13] = 0.2
    img[:, 13:27] = 0.5
    img[:, 27:] = 0.85
    img = np.clip(img + 0.01 * rng.standard_normal((n, n)), 0, 1)
    lab = slic(img, k=16, iters=8)
    checks = [len(np.unique(lab)) >= 8]
    pure = 0
    total = 0
    for u in np.unique(lab):
        mask = lab == u
        stripes = np.zeros(3, int)
        stripes[0] = int((mask & (np.arange(n)[None, :] < 13)).sum())
        stripes[1] = int(
            (mask & ((np.arange(n)[None, :] >= 13) & (np.arange(n)[None, :] < 27))).sum()
        )
        stripes[2] = int((mask & (np.arange(n)[None, :] >= 27)).sum())
        pure += int(stripes.max())
        total += int(mask.sum())
    checks.append(pure / max(1, total) > 0.85)
    sizes = [int((lab == u).sum()) for u in np.unique(lab)]
    checks.append(min(sizes) > 10)
    checks.append(max(sizes) < n * n // 4)
    return {"synthetic_slic_superpixels": float(np.mean(checks))}
