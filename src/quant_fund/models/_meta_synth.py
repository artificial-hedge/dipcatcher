"""Shared few-shot fixture (Finn et al. 2017 sine regression): task = (SYNTHETIC)
A·sin(x + φ), K support points + M query. Metric: query MSE after
adaptation vs a non-meta (pooled) baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sine_task(
    rng: np.random.Generator, K: int = 5, M: int = 15
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    A = rng.uniform(0.1, 5.0)
    phi = rng.uniform(0, np.pi)
    xs = rng.uniform(-5, 5, K)
    xq = rng.uniform(-5, 5, M)
    return (
        xs.astype(np.float64),
        (A * np.sin(xs + phi)).astype(np.float64),
        xq.astype(np.float64),
        (A * np.sin(xq + phi)).astype(np.float64),
    )
