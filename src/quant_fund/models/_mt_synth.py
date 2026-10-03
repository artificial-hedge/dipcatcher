"""Shared SYNTHETIC fixture for wave-178 multi-task-gradient canon:
two conflicting tasks sharing a trunk — task A wants positive
projection, task B orthogonal-ish; losses conflict at the shared
representation. Metric: final min-task acc + mean acc vs naive-sum SGD.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def mt_data(
    seed: int, n: int = 300, d: int = 8
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w1 = rng.standard_normal(d)
    w1 /= np.linalg.norm(w1)
    w2 = rng.standard_normal(d)
    w2 /= np.linalg.norm(w2)
    y1 = (X @ w1 * 2 > 0).astype(np.int64)
    y2 = (X @ w2 * 2 > 0).astype(np.int64)
    return X, y1, y2
