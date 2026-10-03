"""Synthetic fixtures for the w141 certified-robustness canon."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def synth_cls_2d(
    n: int,
    rng: np.random.Generator,
    margin: float = 1.2,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Two classes split by x0+0.5·x1 sign with a clean margin."""
    x = rng.uniform(-2, 2, (n, 4))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int64)
    keep = np.abs(x[:, 0] + 0.5 * x[:, 1]) > 0.15
    return x[keep], y[keep]


def synth_subset(
    n: int,
    m: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.bool_]]:
    """Label depends only on the first k of m dims — true mask returned."""
    x = rng.standard_normal((n, m))
    w = np.zeros(m)
    w[:k] = rng.uniform(0.8, 1.2, k)
    y = (x @ w + 0.1 * rng.standard_normal(n) > 0).astype(np.int64)
    mask = np.zeros(m, dtype=bool)
    mask[:k] = True
    return x, y, mask
