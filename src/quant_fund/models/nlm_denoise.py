"""Non-local means denoising: patch-similarity weighted averaging in a
local search window (SYNTHETIC bench only)."""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 991


def nlm(img: np.ndarray, h: float = 0.1, patch: int = 2, search: int = 5) -> np.ndarray:
    n, m = img.shape
    pad = np.pad(img, patch, mode="reflect")
    patches = np.lib.stride_tricks.sliding_window_view(pad, (2 * patch + 1, 2 * patch + 1))
    out = np.zeros_like(img)
    for i in range(n):
        for j in range(m):
            y0, y1 = max(0, i - search), min(n, i + search + 1)
            x0, x1 = max(0, j - search), min(m, j + search + 1)
            sub = patches[y0:y1, x0:x1]
            d2 = ((sub - patches[i, j]) ** 2).mean(axis=(2, 3))
            w = np.exp(-d2 / (h * h))
            w_sum = float(w.sum())
            if w_sum == 0:
                out[i, j] = img[i, j]
            else:
                out[i, j] = float((w * img[y0:y1, x0:x1]).sum() / w_sum)
    return out


def bench_nlm_denoise(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 24
    img = np.zeros((n, n))
    img[:, :12] = 0.25
    img[:, 12:] = 0.75
    noisy = np.clip(img + 0.15 * rng.standard_normal((n, n)), 0, 1)
    den = nlm(noisy, h=0.12, patch=1, search=3)
    checks = []
    checks.append(float(np.std(den[:, 2:10] - 0.25)) < 0.6 * float(np.std(noisy[:, 2:10] - 0.25)))
    checks.append(float(np.std(den[:, 14:22] - 0.75)) < 0.6 * float(np.std(noisy[:, 14:22] - 0.75)))
    edge_noisy = float(np.mean(np.abs(np.diff(noisy[:, 10:14], axis=1))))
    edge_den = float(np.mean(np.abs(np.diff(den[:, 10:14], axis=1))))
    checks.append(edge_den > 0.6 * edge_noisy)
    flat = np.clip(0.5 + 0.15 * rng.standard_normal((n, n)), 0, 1)
    den2 = nlm(flat, h=0.12, patch=1, search=3)
    checks.append(float(np.std(den2 - 0.5)) < 0.6 * float(np.std(flat - 0.5)))
    return {"synthetic_nlm_denoise": float(np.mean(checks))}
