"""Shared synthetic SSL / test-time-adaptation fixtures (wave 139) (SYNTHETIC).

- `synth_ssl`: 4-class Gaussian blobs; `make_views` produces two noisy
  views (random scale + translation) — view-invariant representations
  separate the classes.
- `synth_tta_split`: train set ID, test set rotated + shifted + rescaled —
  no labels at test; adaptation must recover accuracy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_D = 16
_K = 4


def synth_ssl(n: int, rng: np.random.Generator) -> tuple[FloatArray, NDArray[np.int64]]:
    """XOR-class latent: y = (sign(x0) == sign(x1)) XOR structure — the
    raw features are NOT linearly separable, so a linear probe on x caps
    near chance while a learned representation can expose the class."""
    s1 = rng.choice([-1.0, 1.0], n)
    s2 = rng.choice([-1.0, 1.0], n)
    y = ((s1 * s2) > 0).astype(np.int64)
    x = np.zeros((n, _D))
    x[:, 0] = s1 * 2.0 + 0.4 * rng.standard_normal(n)
    x[:, 1] = s2 * 2.0 + 0.4 * rng.standard_normal(n)
    x[:, 2] = 0.4 * rng.standard_normal(n)
    x[:, 3:] = rng.standard_normal((n, _D - 3))
    return x, y.astype(np.int64)


def make_views(x: FloatArray, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    sc1 = rng.uniform(0.7, 1.3, (x.shape[0], 1))
    sc2 = rng.uniform(0.7, 1.3, (x.shape[0], 1))
    sh1 = rng.normal(0, 0.4, (x.shape[0], _D))
    sh2 = rng.normal(0, 0.4, (x.shape[0], _D))
    return x * sc1 + sh1 + 0.15 * rng.standard_normal(
        x.shape
    ), x * sc2 + sh2 + 0.15 * rng.standard_normal(x.shape)


def synth_tta_split(
    n_tr: int, n_te: int, rng: np.random.Generator
) -> tuple[FloatArray, NDArray[np.int64], FloatArray, NDArray[np.int64]]:
    xtr, ytr = synth_ssl(n_tr, rng)
    xte, yte = synth_ssl(n_te, rng)
    g = rng.standard_normal((_D, _D))
    q, _ = np.linalg.qr(g)
    xte = 1.15 * (xte @ q) + 0.6
    return xtr, ytr, xte, yte


def linear_probe_acc(
    xtr: FloatArray, ytr: NDArray[np.int64], xte: FloatArray, yte: NDArray[np.int64]
) -> float:
    w = np.linalg.lstsq(
        np.hstack([xtr, np.ones((xtr.shape[0], 1))]),
        np.eye(_K)[ytr],
        rcond=None,
    )[0]
    pred = np.argmax(np.hstack([xte, np.ones((xte.shape[0], 1))]) @ w, 1)
    return float((pred == yte).mean())
