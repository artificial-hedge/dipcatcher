"""Canny edge detector: Gaussian smooth, Sobel gradients, non-maximum
suppression, double-threshold hysteresis (SYNTHETIC bench only)."""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 987


def gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    pad = np.pad(img, ((0, 0), (r, r)), mode="reflect")
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, pad)
    pad2 = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, pad2)
    return np.asarray(out, dtype=np.float64)


def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 4.0
    ky = kx.T
    pad = np.pad(img, 1, mode="reflect")
    n, m = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    for i in range(n):
        for j in range(m):
            w = pad[i : i + 3, j : j + 3]
            gx[i, j] = float((w * kx).sum())
            gy[i, j] = float((w * ky).sum())
    return gx, gy


def nonmax_suppress(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    n, m = mag.shape
    out = np.zeros_like(mag)
    deg = np.degrees(ang) % 180
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            a = deg[i, j]
            if a < 22.5 or a >= 157.5:
                n1, n2 = mag[i, j - 1], mag[i, j + 1]
            elif a < 67.5:
                n1, n2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif a < 112.5:
                n1, n2 = mag[i - 1, j], mag[i + 1, j]
            else:
                n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]
            if mag[i, j] >= n1 and mag[i, j] >= n2:
                out[i, j] = mag[i, j]
    return out


def hysteresis(thin: np.ndarray, lo: float, hi: float) -> np.ndarray:
    strong = thin >= hi
    weak = (thin >= lo) & ~strong
    edges = strong.copy()
    stack = [tuple(p) for p in np.argwhere(strong)]
    while stack:
        i, j = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < thin.shape[0]
                    and 0 <= nj < thin.shape[1]
                    and weak[ni, nj]
                    and not edges[ni, nj]
                ):
                    edges[ni, nj] = True
                    stack.append((ni, nj))
    return edges


def canny(img: np.ndarray, sigma: float = 1.0, lo: float = 0.05, hi: float = 0.2) -> np.ndarray:
    sm = gaussian_blur(img, sigma)
    gx, gy = sobel(sm)
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    thin = nonmax_suppress(mag, np.arctan2(gy, gx))
    return hysteresis(thin, lo, hi)


def bench_canny_edge(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 40
    yy, xx = np.mgrid[0:n, 0:n]
    true_edge = np.abs(np.hypot(yy - 20, xx - 20) - 10) < 0.8
    img = (np.hypot(yy - 20, xx - 20) <= 10).astype(float)
    noisy = np.clip(img + 0.08 * rng.standard_normal((n, n)), 0, 1)
    edges = canny(noisy)
    checks = [int(edges.sum()) > 30]
    hits = 0
    tot = int(edges.sum())
    for i, j in np.argwhere(edges):
        if true_edge[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2].any():
            hits += 1
    checks.append(hits / max(1, tot) > 0.8)
    interior = np.zeros((n, n), bool)
    interior[14:26, 14:26] = np.hypot(yy[14:26, 14:26] - 20, xx[14:26, 14:26] - 20) < 8
    checks.append(int((edges & interior).sum()) < tot // 5)
    checks.append(int(canny(np.zeros((n, n))).sum()) == 0)
    return {"synthetic_canny_edge": float(np.mean(checks))}
