"""Polynomial chaos expansion — Legendre gPC on uniform inputs (SYNTHETIC).

Coefficients by Gaussian-quadrature projection; Sobol sensitivity
indices read directly off the expansion (Sobol decomposition is the
orthonormal Fourier decomposition).
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _legendre(n: int, x: FloatArray) -> FloatArray:
    """Normalized Legendre P_n evaluated at x ∈ [−1,1]."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return np.asarray(np.sqrt(3.0) * x)
    p0 = np.ones_like(x)
    p1 = x.copy()
    for k in range(1, n):
        p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
    return np.asarray(np.sqrt(2 * n + 1) * p1)


def _gauss_legendre(m: int) -> tuple[FloatArray, FloatArray]:
    return np.polynomial.legendre.leggauss(m)


def _multi_indices(d: int, deg: int) -> list[tuple[int, ...]]:
    return [a for a in itertools.product(range(deg + 1), repeat=d) if sum(a) <= deg]


def pce_fit(f, d: int, degree: int, n_quad: int = 12) -> tuple[FloatArray, list[tuple[int, ...]]]:
    """Total-degree Legendre PCE on [0,1]^d → coefficients.

    Returns (coefs, multi_indices) with coefs aligned to the basis
    ∏_k ℓ_{a_k}(2x_k − 1)/√(2a_k+1)... (orthonormal on uniform).
    """
    idxs = _multi_indices(d, degree)
    x1, w1 = _gauss_legendre(n_quad)
    xs = 0.5 * (x1 + 1.0)  # map to [0,1]
    ws = w1 / 2.0
    grid = np.array(np.meshgrid(*[xs] * d, indexing="ij")).reshape(d, -1)
    wts = np.prod(np.array(np.meshgrid(*[ws] * d, indexing="ij")).reshape(d, -1), axis=0)
    pts = grid.T
    vals = np.array([f(p) for p in pts])
    # basis matrix
    B = np.ones((pts.shape[0], len(idxs)))
    for j, a in enumerate(idxs):
        b = np.ones(pts.shape[0])
        for k in range(d):
            b *= _legendre(a[k], 2.0 * pts[:, k] - 1.0)
        B[:, j] = b
    coefs = (B * wts[:, None]).T @ vals
    return np.asarray(coefs, dtype=np.float64), idxs


def pce_sobol(coefs: FloatArray, idxs: list[tuple[int, ...]], d: int) -> FloatArray:
    """First-order Sobol indices from PCE coefficients."""
    total_var = float(np.sum(coefs[1:] ** 2))
    out = np.zeros(d)
    for j, a in enumerate(idxs):
        if j == 0 or total_var <= 0:
            continue
        nz = [k for k in range(d) if a[k] > 0]
        if len(nz) == 1:
            out[nz[0]] += coefs[j] ** 2 / total_var
    return out


def bench_pce(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: coefficient convergence on exp(x); Sobol indices on
    the Ishigami function vs. its analytic values."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # E[exp(X)] on [0,1]: e-1 ≈ 1.718
    coefs, idxs = pce_fit(lambda p: float(np.exp(p[0])), 1, 6)
    out["synthetic_pce_mean_err"] = abs(float(coefs[0]) - (np.e - 1))
    var_est = float(np.sum(coefs[1:] ** 2))
    var_true = 0.5 * (np.e**2 - 1) - (np.e - 1) ** 2
    out["synthetic_pce_var_err"] = abs(var_est - var_true)
    # Ishigami: f = sin(x1) + a sin²(x2) + b x3⁴ sin(x1), a=7, b=0.1
    a, b = 7.0, 0.1

    def ish(p: FloatArray) -> float:
        x = 2.0 * np.pi * np.asarray(p) - np.pi  # uniform on [-π, π]³
        return float(np.sin(x[0]) + a * np.sin(x[1]) ** 2 + b * x[2] ** 4 * np.sin(x[0]))

    c2, i2 = pce_fit(ish, 3, 6)
    sob = pce_sobol(c2, i2, 3)
    # analytic first-order indices for the standard Ishigami params
    # S1 ≈ 0.314, S2 ≈ 0.442, S3 ≈ 0.0 (first order!)
    out["synthetic_pce_sobol"] = sob.tolist()[0]
    out["synthetic_pce_sobol_err"] = float(np.linalg.norm(sob - np.array([0.314, 0.442, 0.0])))
    out["synthetic_pce_sobol_s3_zero"] = float(sob[2] < 0.02)
    out["synthetic_pce_ranking_ok"] = float(sob[1] > sob[0] > sob[2])
    del rng
    return out


if __name__ == "__main__":
    print(bench_pce())
