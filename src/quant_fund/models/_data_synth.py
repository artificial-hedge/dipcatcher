"""Synthetic data-dynamics fixture (SYNTHETIC).

2-class problem in R^8: class mean ±mu on first 3 dims; dim 3 is a
spurious weakly label-correlated feature; dims 4–7 are distractors
amplified ×4 on hard examples (easy/hard split). y = sign(x·w_true) on
the first 3 dims; hardness score = distractor magnitude.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def synth_dataset(n: int, rng: np.random.Generator, hard_frac: float = 0.3):
    w = np.zeros(8)
    w[:3] = np.array([1.0, 0.8, 0.6])
    y = rng.integers(0, 2, n)
    x = rng.normal(0, 1, (n, 8))
    x[:, :3] += (2 * y - 1)[:, None] * w[:3][None, :] * 0.25
    hard = rng.random(n) < hard_frac
    x[hard, 3:] *= 4.0  # distractor noise ×4 on hard examples
    # shortcut feature: dim 3 weakly label-correlated (spurious)
    spurious = rng.random(n) < 0.9
    x[spurious, 3] = (2 * y[spurious] - 1) * 1.5 + rng.normal(0, 0.2, spurious.sum())
    return x.astype(np.float64), y, hard


def true_margin(x: FloatArray, y: IntArray) -> FloatArray:
    w = np.array([1.0, 0.8, 0.6] + [0] * 5)
    return np.asarray((x @ w) * (2 * y - 1))
