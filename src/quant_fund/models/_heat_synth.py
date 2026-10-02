"""Shared fixture for wave-187 scientific-ML/PDE-solver canon.

1-D heat equation u_t = (sigma²/2) u_xx on [-4,4]×[0,T] with
Gaussian initial condition u0(x)=N(x; mu0, s0²). Analytic solution:
Gaussian with variance s0² + sigma²·t. Boundary: Dirichlet (u≈0).
Metric: relative L2 error vs analytic solution on a grid.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

L, T_, SIG, MU0, S0 = 4.0, 0.5, 0.7, 0.0, 0.4


def grid(nx: int = 101) -> FloatArray:
    return np.asarray(np.linspace(-L, L, nx), dtype=np.float64)


def u0(x: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-0.5 * (x - MU0) ** 2 / S0**2) / (S0 * np.sqrt(2 * np.pi)))


def u_exact(x: FloatArray, t: float) -> FloatArray:
    v = S0**2 + SIG**2 * t
    return np.asarray(np.exp(-0.5 * (x - MU0) ** 2 / v) / np.sqrt(2 * np.pi * v))


def rel_l2(pred: FloatArray, truth: FloatArray) -> float:
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def eval_error(pred_fn: Callable[[FloatArray, float], FloatArray]) -> float:
    x = grid()
    return rel_l2(pred_fn(x, T_), u_exact(x, T_))


def mc_paths(n: int, t: float, seed: int) -> FloatArray:
    """Feynman-Kac helper: terminal positions of BM from x0=u0-sampled."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(MU0, S0, n)
    return np.asarray(x0 + SIG * np.sqrt(t) * rng.standard_normal(n))
