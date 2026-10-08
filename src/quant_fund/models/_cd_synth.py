"""Shared conditional-density fixture: multimodal target — sign branch (SYNTHETIC)
switches on x0; heteroscedastic noise. Score: test log-density vs
single-Gaussian baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cd_data(
    seed: int = 7, n: int = 400
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (n, 3))
    branch = np.sign(x[:, 0])
    y = branch * np.sin(2 * x[:, 1]) + 0.1 * (1 + np.abs(x[:, 2])) * rng.standard_normal(n)
    x_te = rng.uniform(-2, 2, (n // 2, 3))
    b_te = np.sign(x_te[:, 0])
    y_te = b_te * np.sin(2 * x_te[:, 1]) + 0.1 * (1 + np.abs(x_te[:, 2])) * rng.standard_normal(
        n // 2
    )
    return x, y, x_te, y_te


def gauss_logpdf(y: NDArray[np.float64], mu: float, sd: float) -> NDArray[np.float64]:
    return np.asarray(-0.5 * np.log(2 * np.pi * sd**2) - (y - mu) ** 2 / (2 * sd**2))
