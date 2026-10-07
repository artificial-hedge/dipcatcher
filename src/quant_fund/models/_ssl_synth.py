"""Shared synthetic SSL / test-time-adaptation fixtures (wave 139) (SYNTHETIC).

- `synth_ssl`: 2-class XOR latent (features not linearly separable);
  `make_views` produces two noisy views (random scale + translation) —
  view-invariant representations separate the classes.
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
    if n < 1:
        raise ValueError(f"need n>=1, got {n}")
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
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != _D:
        raise ValueError(f"x must be a non-empty (n, {_D}) array, got {x.shape}")
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
    if n_tr < 1 or n_te < 1:
        raise ValueError(f"need n_tr>=1 and n_te>=1, got {n_tr},{n_te}")
    xtr, ytr = synth_ssl(n_tr, rng)
    xte, yte = synth_ssl(n_te, rng)
    g = rng.standard_normal((_D, _D))
    q, r_ = np.linalg.qr(g)
    s_ = np.sign(np.diag(r_))
    s_[s_ == 0] = 1.0
    q = q * s_[None, :]  # canonicalize QR sign (LAPACK-arbitrary)
    xte = 1.15 * (xte @ q) + 0.6
    return xtr, ytr, xte, yte


def linear_probe_acc(
    xtr: FloatArray, ytr: NDArray[np.int64], xte: FloatArray, yte: NDArray[np.int64]
) -> float:
    if (
        xtr.ndim != 2
        or xte.ndim != 2
        or xtr.shape[0] < 1
        or xte.shape[0] < 1
        or xtr.shape[1] != xte.shape[1]
    ):
        raise ValueError(f"need matching non-empty (n,d) arrays, got {xtr.shape} vs {xte.shape}")
    if len(ytr) != xtr.shape[0] or len(yte) != xte.shape[0]:
        raise ValueError("label arrays must match row counts")
    if ytr.min() < 0 or ytr.max() >= _K or yte.min() < 0 or yte.max() >= _K:
        raise ValueError(f"labels must be in [0, {_K})")
    w = np.linalg.lstsq(
        np.hstack([xtr, np.ones((xtr.shape[0], 1))]),
        np.eye(_K)[ytr],
        rcond=None,
    )[0]
    pred = np.argmax(np.hstack([xte, np.ones((xte.shape[0], 1))]) @ w, 1)
    return float((pred == yte).mean())
