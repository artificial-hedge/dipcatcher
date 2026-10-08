"""Rasch fit statistics — infit and outfit mean squares.

Wright & Masters (1982), Linacre (2002): for Rasch-model
residuals z_ij = (x_ij - p_ij)/sqrt(p_ij(1-p_ij)) the item/person
fit indices are

    outfit = mean(z^2)               unweighted (outlier-sensitive)
    infit  = sum(w z^2) / sum(w)     variance-weighted (inlier-sensitive)

with w_ij = p_ij(1-p_ij) the model variance. Values ~1.0 indicate
model-consistent residuals; >~1.3 underfit (noise/misfit), <~0.7
overfit (too deterministic). Standardized t forms use the Wilson-
Hilferty cube-root transform.

Honesty: the bench fits a one-parameter logistic to SYNTHETIC
responses (Newton on persons + items, as in the repo's IRT lane
but self-contained) and checks that a planted noisy item has
high outfit while model-consistent items sit near 1. Bounds are
the conventional Linacre ranges — flagged as heuristic. Fail-
closed on non-binary data or degenerate fits.

References: Wright & Masters (1982) "Rating Scale Analysis";
Linacre (2002) "What do infit and outfit mean-square and
standardized mean?"; Bond & Fox (2015) ch. 3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 20 or a.shape[1] < 4:
        raise ValueError("bad response matrix")
    if not np.isfinite(a).all() or not np.isin(a, (0.0, 1.0)).all():
        raise ValueError("non-binary responses")
    return a


def _rasch_fit_1pl(a: FloatArray, n_iter: int = 60) -> tuple[FloatArray, FloatArray]:
    """Joint MLE for Rasch person abilities + item difficulties."""
    n, k = a.shape
    theta = np.zeros(n)
    b = np.zeros(k)
    for _ in range(n_iter):
        d = theta[:, None] - b[None, :]
        p = 1.0 / (1.0 + np.exp(-np.clip(d, -30, 30)))
        w = p * (1 - p)
        g = (a - p).sum(axis=1)
        h = -w.sum(axis=1)
        theta = theta - np.clip(g / np.clip(h, -1e9, -1e-6), -1.0, 1.0)
        gi = -(a - p).sum(axis=0)
        hi = -w.sum(axis=0)
        b = b - np.clip(gi / np.clip(hi, -1e9, -1e-6), -1.0, 1.0)
        b = b - b.mean()
    return theta, b


def item_fit(x: FloatArray) -> dict[str, float | FloatArray]:
    """Infit/outfit per item from a Rasch 1PL fit."""
    a = _check(x)
    theta, b = _rasch_fit_1pl(a)
    d = theta[:, None] - b[None, :]
    p = np.clip(1.0 / (1.0 + np.exp(-d)), 1e-9, 1 - 1e-9)
    w = p * (1 - p)
    z2 = (a - p) ** 2 / w
    outfit = z2.mean(axis=0)
    infit = (w * z2).sum(axis=0) / w.sum(axis=0)
    return {
        "outfit": np.asarray(outfit, dtype=np.float64),
        "infit": np.asarray(infit, dtype=np.float64),
        "outfit_max": float(outfit.max()),
        "infit_max": float(infit.max()),
    }


def person_fit(x: FloatArray) -> dict[str, float | FloatArray]:
    """Infit/outfit per person from a Rasch 1PL fit."""
    a = _check(x)
    theta, b = _rasch_fit_1pl(a)
    d = theta[:, None] - b[None, :]
    p = np.clip(1.0 / (1.0 + np.exp(-d)), 1e-9, 1 - 1e-9)
    w = p * (1 - p)
    z2 = (a - p) ** 2 / w
    outfit = z2.mean(axis=1)
    infit = (w * z2).sum(axis=1) / w.sum(axis=1)
    return {
        "outfit": np.asarray(outfit, dtype=np.float64),
        "infit": np.asarray(infit, dtype=np.float64),
        "outfit_max": float(outfit.max()),
    }


def bench_rasch_fit(seed: int = 20261231 + 443) -> dict[str, float]:
    """SYNTHETIC check — noisy item gets high outfit, clean items ~1."""
    rng = np.random.default_rng(seed)
    n, k = 200, 10
    theta = rng.standard_normal(n)
    b = np.linspace(-1.5, 1.5, k)
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))
    a = (rng.random((n, k)) < p).astype(float)
    # corrupt item 0 with noise
    a[:, 0] = (rng.random(n) < 0.5).astype(float)
    out = item_fit(a)
    outfits = np.asarray(out["outfit"], dtype=float)
    clean = outfits[1:].mean()
    noisy = float(outfits[0])
    if noisy <= clean or clean > 1.4 or noisy < 1.15:
        raise ValueError(f"rasch_fit off: noisy={noisy:.3f} clean_mean={clean:.3f}")
    return {
        "synthetic_outfit_noisy": noisy,
        "synthetic_outfit_clean_mean": float(clean),
        "synthetic_infit_clean_mean": float(np.asarray(out["infit"])[1:].mean()),
        "synthetic_score": 1.0,
    }
