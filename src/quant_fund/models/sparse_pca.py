"""Sparse PCA canon: l1-penalized variance maximization via alternating (SYNTHETIC)
maximization (power iteration with soft-thresholded loadings, SCoTLASS
style), with Schur-complement deflation for multi-component fits.
``bench_sparse_pca`` plants a block-sparse covariance and gates support
recovery of the leading sparse direction plus angle to the planted
component.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _soft(x: FloatArray, lam: float) -> FloatArray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def spca_direction(cov: FloatArray, lam: float, it: int = 200, seed: int = 0) -> FloatArray:
    """Single sparse leading direction: alternating y = Sigma x / |.|,
    x = soft(y, lam) / |.| (variance maximization under l1)."""
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    rng = np.random.default_rng(seed)
    x: FloatArray = np.asarray(rng.standard_normal(d), dtype=np.float64)
    x = np.asarray(x / float(np.linalg.norm(x)), dtype=np.float64)
    for _ in range(it):
        y = cov @ x
        ny = np.linalg.norm(y)
        if ny <= 0:
            break
        x_new: FloatArray = _soft(y / ny, lam)
        nx = float(np.linalg.norm(x_new))
        if nx <= 1e-12:
            x_new = y / ny
            nx = 1.0
        x_new = x_new / nx
        if np.max(np.abs(x_new - x)) < 1e-10:
            x = x_new
            break
        x = x_new
    return np.asarray(x, dtype=np.float64)


def spca_fit(
    cov: FloatArray, lam: float, n_comp: int, seed: int = 0
) -> tuple[FloatArray, FloatArray]:
    """Returns (components (k,d), explained variance per comp) via
    Schur-complement deflation."""
    cov = np.asarray(cov, dtype=np.float64).copy()
    comps = np.zeros((n_comp, cov.shape[0]))
    var = np.zeros(n_comp)
    resid = cov.copy()
    for k in range(n_comp):
        v = spca_direction(resid, lam, seed=seed + k + 1)
        comps[k] = v
        var[k] = float(v @ cov @ v)
        resid = resid - np.outer(resid @ v, v)
    return comps, var


def spca_support(v: FloatArray, tol: float = 1e-3) -> IntArray:
    return np.asarray(np.flatnonzero(np.abs(v) > tol), dtype=np.int64)


def bench_sparse_pca(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = 30
    # planted sparse direction on first 5 coords + rank-1 background
    v_true = np.zeros(d)
    v_true[:5] = np.array([0.5, 0.5, 0.5, 0.5, 0.2])
    v_true /= np.linalg.norm(v_true)
    b = rng.standard_normal(d)
    b /= np.linalg.norm(b)
    cov = 3.0 * np.outer(v_true, v_true) + 0.6 * np.outer(b, b) + 0.1 * np.eye(d)
    comps, var = spca_fit(cov, lam=0.08, n_comp=2, seed=seed)
    v1 = comps[0]
    if v1 @ v_true < 0:
        v1 = -v1
    supp = spca_support(v1)
    dense = np.linalg.eigh(cov)[1][:, -1]
    dense_supp = spca_support(dense)
    return {
        "synthetic_spca_angle": float(abs(v1 @ v_true)),
        "synthetic_spca_support_hit": float(np.intersect1d(supp, np.arange(5)).size),
        "synthetic_spca_support_extra": float(np.setdiff1d(supp, np.arange(5)).size),
        "synthetic_dense_support_size": float(dense_supp.size),
        "synthetic_spca_var_ratio": float(var[0] / (var[0] + var[1])),
    }
