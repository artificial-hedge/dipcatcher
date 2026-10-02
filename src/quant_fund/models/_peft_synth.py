"""Synthetic fixtures for the w143 PEFT canon."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def synth_peft_base(n: int, rng: np.random.Generator) -> tuple[FloatArray, NDArray[np.int64]]:
    """Base task: linear boundary in R^4."""
    x = rng.uniform(-2, 2, (n, 4))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int64)
    return x, y


def synth_peft_shift(
    n: int, rng: np.random.Generator, rot: float = 0.6
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Adaptation task: boundary rotated by `rot` radians."""
    x = rng.uniform(-2, 2, (n, 4))
    c, s = np.cos(rot), np.sin(rot)
    y = (c * x[:, 0] + s * x[:, 1] + 0.5 * x[:, 2] > 0).astype(np.int64)
    return x, y
