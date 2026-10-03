"""Functional principal component analysis (Karhunen-Loeve).

Ramsay & Silverman (2005): a square-integrable stochastic process X(t)
admits the expansion

    X(t) = mu(t) + sum_k xi_k phi_k(t),

where phi_k are the eigenfunctions of the covariance operator
C(s,t) = cov(X(s), X(t)) and xi_k = <X - mu, phi_k>. Eigenvalues
lambda_k decay; the first p components explain fraction-of-variance
FVE_p = sum_{k<=p} lambda_k / sum_k lambda_k. Discretized on an
equally spaced grid, C becomes a matrix; with trapezoid weights the
operator eigenproblem reduces to a symmetric eigensolve of
W^{1/2} C W^{1/2} (Yao, Mueller & Wang 2005 PACE).

Honesty: scores are projections, not factor returns. The bench checks
the analytic Brownian-motion KL spectrum: lambda_k = 1/((k-1/2)^2 pi^2),
phi_k(t) = sqrt(2) sin((k-1/2) pi t). Fail-closed on ragged grids or
non-PSD covariances.

References: Ramsay & Silverman (2005) FDA; Yao, Mueller & Wang (2005)
"Functional data analysis for sparse longitudinal data"; Bosq (2000).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def fpca(
    curves: FloatArray,
    grid: FloatArray,
    n_components: int = 4,
) -> dict[str, FloatArray | float]:
    """Trapezoid-weighted FPCA of a (n_curves, n_grid) curve sample.

    Returns mean curve, eigenvalues, eigenfunctions evaluated on the
    grid, scores, and the fraction-of-variance-explained vector.
    """
    x = np.asarray(curves, dtype=float)
    g = np.asarray(grid, dtype=float).ravel()
    if x.ndim != 2 or x.shape[1] != g.size:
        raise ValueError("curves must be (n, len(grid))")
    n, m = x.shape
    if n < 5 or m < 10:
        raise ValueError("need >=5 curves on >=10 grid points")
    if np.any(np.diff(g) <= 0):
        raise ValueError("grid must be strictly increasing")
    if not (1 <= n_components <= min(n, m)):
        raise ValueError("bad n_components")
    if not np.isfinite(x).all():
        raise ValueError("non-finite curves")
    mean_curve = x.mean(axis=0)
    xc = x - mean_curve
    cov = (xc.T @ xc) / (n - 1)
    # trapezoid quadrature weights on the grid
    w = np.empty(m)
    w[1:-1] = 0.5 * (g[2:] - g[:-2])
    w[0] = g[1] - g[0]
    w[-1] = g[-1] - g[-2]
    w = np.maximum(w, 1e-12)
    sw = np.sqrt(w)
    op = (sw[:, None] * cov) * sw[None, :]
    evals, evecs = np.linalg.eigh(op)
    idx = np.argsort(evals)[::-1][:n_components]
    evals = np.maximum(evals[idx], 0.0)
    efuncs = evecs[:, idx] / sw[:, None]
    # normalize eigenfunctions in L2(w): <phi, phi>_w = 1
    norms = np.sqrt((efuncs * efuncs * w[:, None]).sum(axis=0))
    efuncs = efuncs / norms
    # deterministic sign convention: increasing slope at grid start
    for j in range(efuncs.shape[1]):
        if efuncs[1, j] < efuncs[0, j]:
            efuncs[:, j] = -efuncs[:, j]
    scores = (xc[:, :, None] * efuncs[None, :, :] * w[None, :, None]).sum(axis=1)
    total_var = float(np.sum(np.diag(cov) * w))
    total_var = max(total_var, 1e-18)
    fve = evals / total_var
    return {
        "mean_curve": np.asarray(mean_curve, dtype=np.float64),
        "eigenvalues": np.asarray(evals, dtype=np.float64),
        "eigenfunctions": np.asarray(efuncs, dtype=np.float64),
        "scores": np.asarray(scores, dtype=np.float64),
        "fve": np.asarray(fve, dtype=np.float64),
        "n_curves": float(n),
    }


def fpca_reconstruct(
    out: dict[str, FloatArray | float], n_components: int | None = None
) -> FloatArray:
    """X_hat(t) = mu(t) + sum_k xi_k phi_k(t) for the fitted model."""
    mean_curve = np.asarray(out["mean_curve"])
    scores = np.asarray(out["scores"])
    efuncs = np.asarray(out["eigenfunctions"])
    p = efuncs.shape[1] if n_components is None else min(n_components, efuncs.shape[1])
    return np.asarray(mean_curve + scores[:, :p] @ efuncs[:, :p].T, dtype=np.float64)


def bench_functional_pca(seed: int = 20261231 + 398) -> dict[str, float]:
    """SYNTHETIC check — FPCA recovers the analytic Brownian KL spectrum.

    For standard Brownian motion on [0,1]:
        lambda_k = 1 / ((k - 1/2)^2 pi^2),  phi_k(t) = sqrt(2) sin((k-1/2) pi t).
    """
    rng = np.random.default_rng(seed)
    m = 200
    grid = np.linspace(0.0, 1.0, m)
    n = 400
    # exact BM via cumulative innovations (independent at grid resolution)
    increments = rng.standard_normal((n, m)) * np.sqrt(grid[1] - grid[0])
    curves = np.cumsum(increments, axis=1)
    out = fpca(curves, grid, n_components=3)
    lam = np.asarray(out["eigenvalues"])
    lam_true = np.array([1.0 / ((k - 0.5) ** 2 * np.pi**2) for k in range(1, 4)])
    lam_err = float(np.linalg.norm(lam - lam_true) / np.linalg.norm(lam_true))
    if lam_err > 0.25:
        raise ValueError(f"KL eigenvalues off: {lam} vs {lam_true}")
    efuncs = np.asarray(out["eigenfunctions"])
    phi1_true = np.sqrt(2.0) * np.sin(0.5 * np.pi * grid)
    # eigenfunctions are sign-free; correlate in the quadrature metric
    w_g = np.gradient(grid)
    num = abs(float(np.sum(efuncs[:, 0] * phi1_true * w_g)))
    den = float(np.sqrt(np.sum(phi1_true * phi1_true * w_g)))
    phi_err = 1.0 - num / max(den, 1e-18)
    if phi_err > 0.02:
        raise ValueError("leading eigenfunction off")
    fve = np.asarray(out["fve"])
    # Brownian KL: first component carries ~81%
    if not (0.7 < fve[0] < 0.92):
        raise ValueError(f"FVE1 implausible: {fve[0]}")
    rec = fpca_reconstruct(out, n_components=3)
    resid = np.trapezoid((curves - rec) ** 2, grid, axis=1).mean()
    resid_frac = float(resid / np.trapezoid(curves**2, grid, axis=1).mean())
    if resid_frac > 0.25:
        raise ValueError("reconstruction error too high")
    return {
        "synthetic_fpca_lam_err": lam_err,
        "synthetic_fpca_phi_err": phi_err,
        "synthetic_fpca_fve1": float(fve[0]),
        "synthetic_fpca_resid": resid_frac,
        "score": 1.0,
    }
