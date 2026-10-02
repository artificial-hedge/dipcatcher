"""Bayesian quadrature — GP probabilistic integration.

Square-exponential kernel on [0,1]^d against the uniform measure has
closed-form kernel means: posterior mean/variance of ∫f follow from
the GP posterior over evaluations. Includes WSABI-style moment
matching-free baseline (vanilla BQ).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

FloatArray = NDArray[np.float64]


def _se_kernel(x: FloatArray, y: FloatArray, ls: float, sf: float) -> FloatArray:
    d2 = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    return sf**2 * np.exp(-0.5 * d2 / ls**2)


def _kernel_mean(x: FloatArray, ls: float, sf: float) -> FloatArray:
    """∫ k(x, x') dx' over x' ∈ [0,1]^d — product of 1-D erf terms."""
    c = np.sqrt(np.pi / 2) * ls
    per_dim = erf(x / (np.sqrt(2) * ls)) + erf((1.0 - x) / (np.sqrt(2) * ls))
    out: FloatArray = sf**2 * c ** x.shape[1] * np.prod(per_dim, axis=1)
    return out


def _kernel_kernel_mean(ls: float, sf: float, d: int) -> float:
    """∫∫ k(x, x') dx dx' over the unit cube — closed form SE."""
    one_d = 2.0 * (
        ls * np.sqrt(np.pi / 2) * erf(1.0 / (np.sqrt(2) * ls))
        - ls * ls * (1.0 - np.exp(-0.5 / ls**2))
    )
    return float(sf**2 * one_d**d)


def bayesian_quadrature(
    f,
    d: int,
    n: int,
    ls: float = 0.2,
    sf: float = 1.0,
    seed: int = 0,
) -> tuple[float, float, FloatArray]:
    """Integrate f on [0,1]^d by BQ.

    Returns (posterior_mean, posterior_std, eval_points).
    """
    rng = np.random.default_rng(seed)
    # Latin-hypercube design: stratified per-dimension, shuffled
    X = np.empty((n, d))
    for k in range(d):
        X[:, k] = (rng.permutation(n) + rng.random(n)) / n
    y = np.array([f(x) for x in X])
    K = _se_kernel(X, X, ls, sf) + np.eye(n) * 1e-10
    q = _kernel_mean(X, ls, sf)
    Kinv = np.linalg.inv(K)
    mean = float(q @ Kinv @ y)
    var = _kernel_kernel_mean(ls, sf, d) - float(q @ Kinv @ q)
    return mean, float(np.sqrt(max(var, 0.0))), X


def bench_bayesian_quadrature(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: ∫cos and a smooth product on [0,1]^d — posterior mean
    converges superlinearly vs. MC at the same budget."""
    out: dict[str, float] = {}
    mean, std, _ = bayesian_quadrature(
        lambda x: float(np.cos(np.pi * x[0] / 2)), 1, 6, ls=0.6, seed=seed
    )
    out["synthetic_bq_cos_err"] = abs(mean - 2.0 / np.pi)
    out["synthetic_bq_cos_std"] = std

    # d=3 smooth function: ∫∫∫ sin(Σx) = known
    def f3(x: FloatArray) -> float:
        return float(np.sin(x[0] + x[1] + x[2]))

    exact3 = (
        3.0 * np.cos(1.0) * (1.0 - np.cos(1.0))
        + np.cos(3.0) * (-1)
        + np.cos(2.0) * 3.0 * (-0.5) * 0
        + 0
    )
    # ∫∫∫ sin(x+y+z) over [0,1]^3 = Im[ ∫∫∫ e^{iΣx} ] = Im[(∫e^{ix}dx)^3]
    i1 = (np.e**1j - 1) / 1j
    exact3 = float(np.imag(i1**3))
    mean3, std3, _ = bayesian_quadrature(f3, 3, 20, ls=0.6, seed=seed)
    out["synthetic_bq_sin3_err"] = abs(mean3 - exact3)
    out["synthetic_bq_sin3_std"] = std3
    # MC comparison at same budget
    rng = np.random.default_rng(seed)
    mc = float(np.mean([f3(x) for x in rng.random((20, 3))]))
    out["synthetic_bq_mc_err"] = abs(mc - exact3)
    out["synthetic_bq_beats_mc"] = float(out["synthetic_bq_sin3_err"] < out["synthetic_bq_mc_err"])
    return out


if __name__ == "__main__":
    print(bench_bayesian_quadrature())
