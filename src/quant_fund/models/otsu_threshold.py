"""Otsu threshold selection: maximize between-class variance over the
intensity histogram (SYNTHETIC bench only)."""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 988


def otsu(img: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(img.ravel(), bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(1, img.size)
    w = p.cumsum()
    centers = (edges[:-1] + edges[1:]) / 2
    mu = (p * centers).cumsum()
    mu_t = mu[-1]
    sigma_b2 = np.zeros(bins)
    mid = (w > 0) & (w < 1)
    sigma_b2[mid] = (mu_t * w[mid] - mu[mid]) ** 2 / (w[mid] * (1 - w[mid]))
    best = float(sigma_b2.max())
    plateau = np.flatnonzero(sigma_b2 >= best - 1e-12)
    t = float(centers[int(plateau[len(plateau) // 2])])
    return t


def segment(img: np.ndarray, t: float) -> np.ndarray:
    return img > t


def bench_otsu_threshold(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 60
    img = np.where(
        rng.random((n, n)) < 0.4,
        0.15 + 0.08 * rng.standard_normal((n, n)),
        0.8 + 0.1 * rng.standard_normal((n, n)),
    )
    img = np.clip(img, 0, 1)
    true_fg = img > 0.5
    t = otsu(img)
    checks = [0.3 < t < 0.65]
    seg = segment(img, t)
    err = float(np.mean(seg != true_fg))
    checks.append(err < 0.08)
    dark = np.clip(0.1 + 0.05 * rng.standard_normal((n, n)), 0, 1)
    bright = np.clip(0.85 + 0.05 * rng.standard_normal((n, n)), 0, 1)
    t2 = otsu(np.concatenate([dark.ravel(), bright.ravel()]).reshape(2 * n, n))
    checks.append(0.35 < t2 < 0.65)
    uni = np.full((n, n), 0.5) + 1e-4 * rng.standard_normal((n, n))
    t3 = otsu(np.clip(uni, 0, 1))
    checks.append(0.2 < t3 < 0.8)
    return {"synthetic_otsu_threshold": float(np.mean(checks))}
