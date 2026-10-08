"""Synthetic multivariate series with injected anomalies for the (SYNTHETIC)
anomaly-detection canon: AR-driven normals + sparse point/contextual
outliers with known labels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def anom_series(
    seed: int = 7,
    n: int = 400,
    d: int = 3,
    n_anom: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    # AR(1) latent driving d observed dims
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.9 * z[t - 1] + rng.standard_normal() * 0.3
    load = rng.standard_normal((d, d)) / np.sqrt(d)
    x = z[:, None] * load[0] + rng.standard_normal((n, d)) * 0.2
    y = np.zeros(n, dtype=int)
    idx = rng.choice(n - 20, n_anom, replace=False) + 10
    for i in idx:
        if rng.uniform() < 0.5:
            x[i] += rng.standard_normal(d) * 4  # point anomaly
        else:
            j = min(i + 4, n)
            x[i:j] = x[i:j][::-1] * -1.5  # contextual block
            y[i:j] = 1
            continue
        y[i] = 1
    return x.astype(np.float64), y


def windows(x: NDArray[np.float64], w: int = 16) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return (n-w+1, w, d) windows + index of each window's center."""
    n, d = x.shape
    out = np.stack([x[t : t + w] for t in range(n - w + 1)])
    ctr = np.arange(n - w + 1) + w // 2
    return out.astype(np.float64), ctr


def auc(scores: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    """Rank AUC: P(score_anom > score_normal) with ties = 0.5."""
    a = scores[y == 1]
    b = scores[y == 0]
    if len(a) == 0 or len(b) == 0:
        return 0.5
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)
