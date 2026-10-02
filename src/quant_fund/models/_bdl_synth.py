"""Shared Bayesian-DL fixture: heteroscedastic regression + a held-out
OOD region (|x|>2.5 not in training). Score: NLL/coverage on test + OOD
uncertainty gap (var at OOD vs ID — higher is better).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bdl_data(
    seed: int = 7, n: int = 256, d: int = 4
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (n, d))
    beta = rng.standard_normal(d)
    y = x @ beta + 0.1 * np.abs(x[:, 0]) * rng.standard_normal(n)
    x_te = rng.uniform(-2, 2, (n // 2, d))
    y_te = x_te @ beta + 0.1 * np.abs(x_te[:, 0]) * rng.standard_normal(n // 2)
    x_ood = rng.uniform(2.5, 4, (n // 4, d)) * np.sign(rng.standard_normal((n // 4, d)))
    return x, y, x_te, y_te, x_ood


def nll_gauss(y: NDArray[np.float64], mu: NDArray[np.float64], var: NDArray[np.float64]) -> float:
    var = np.maximum(var, 1e-8)
    return float(0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var).mean())


def coverage(
    y: NDArray[np.float64], mu: NDArray[np.float64], sd: NDArray[np.float64], z: float = 1.96
) -> float:
    return float((np.abs(y - mu) <= z * sd).mean())
