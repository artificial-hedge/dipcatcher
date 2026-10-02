"""Shared SYNTHETIC fixture for wave-175 LM-components canon:
induction-head task — random token sequences where a marked key repeats;
the target is the token following the key's first occurrence. Tests
attention mechanisms' content-based retrieval. Also a multi-regime
density task for MoE routing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

VOCAB = 16


def recall_batch(
    seed: int, B: int = 32, T: int = 12
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """x: (B,T) tokens; y: (B,) value token after first key occurrence."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, VOCAB, (B, T))
    key = rng.integers(1, VOCAB, (B,))
    val = rng.integers(1, VOCAB, (B,))
    y = np.zeros(B, dtype=np.int64)
    for b in range(B):
        i1 = int(rng.integers(0, T - 6))
        i2 = int(rng.integers(i1 + 2, T - 1))
        x[b, i1] = key[b]
        x[b, i1 + 1] = val[b]
        x[b, i2] = key[b]
        y[b] = val[b]
    return x, y


def regime_task(seed: int, n: int = 600, k: int = 4, d: int = 8) -> tuple[FloatArray, FloatArray]:
    """n samples, k regimes: y = sigmoid(x @ w_r) with regime-dependent w."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((k, d))
    W = W / np.linalg.norm(W, axis=1, keepdims=True) * 3.0
    r = rng.integers(0, k, n)
    X = rng.standard_normal((n, d))
    s = (X * W[r]).sum(-1)
    # clean decision boundary + label-flip noise
    y = (s + 0.4 * rng.standard_normal(n) > 0).astype(np.float64)
    return X, y
