"""Vision fixture: 6x6 grayscale "images" (SYNTHETIC) — two corner blobs
(top-left 2x2, bottom-right 2x2); class = XOR of blob presence, so the
task is not linearly separable; plus Gaussian noise. A CNN's locality
helps; patch-token ViT needs more data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def synth_images(
    seed: int = 0, n: int = 400, noise: float = 0.35
) -> tuple[FloatArray, NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, (n, 6, 6))
    y = rng.integers(0, 2, n)
    for i in range(n):
        a = rng.integers(0, 2)
        b = rng.integers(0, 2)
        if a:
            x[i, 0:2, 0:2] += 1.4  # top-left blob
        if b:
            x[i, 4:6, 4:6] += 1.4  # bottom-right blob
        y[i] = a ^ b  # XOR — not linearly separable
    return x.astype(np.float64), y


def patches(x: FloatArray, p: int = 2) -> FloatArray:
    """(n,6,6) -> (n, (6//p)^2, p*p) flattened p x p patches."""
    if 6 % p:
        raise ValueError(f"patch size p={p} does not divide image side 6")
    n = x.shape[0]
    out = np.zeros((n, (6 // p) ** 2, p * p))
    k = 0
    for r in range(0, 6, p):
        for c in range(0, 6, p):
            out[:, k] = x[:, r : r + p, c : c + p].reshape(n, p * p)
            k += 1
    return out
