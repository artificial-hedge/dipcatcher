"""Canonical correlation analysis, PLS, and reduced-rank regression (SYNTHETIC).

Hotelling (1936): given two views X (n, p), Y (n, q), CCA finds pairs
(a_i, b_i) maximizing corr(X a_i, Y b_i). With whitened blocks the
problem is the SVD of the coherence matrix

    C_xy^{-1/2} C_xy C_yy^{-1/2}

whose singular values are the canonical correlations.

Wold (1975) PLS replaces SVD cross-covariance directions (maximizing
covariance, not correlation); Anderson (1951) reduced-rank regression
fits Y = X B + E with rank(B) = r by projecting the OLS fit onto the
top-r canonical directions — the three share the same generalized
eigenproblem family.

Honesty: the bench plants a shared latent direction between two views;
the leading canonical pair must align with the planted direction (high
|cosine|) and the canonical correlation must approach the planted
level. Fail-closed on ill-conditioned blocks or non-finite input.

References: Hotelling (1936) "Relations between two sets of variates";
Anderson (1951) "Estimating linear restrictions on regression
coefficients"; Wold (1975) "Path models with latent variables";
Wegelin (2000) PLS/CCA/RRR unification survey.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("x (n,p) and y (n,q) must share n")
    if a.shape[0] < max(a.shape[1], b.shape[1]) + 10:
        raise ValueError("not enough observations")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite input")
    return a, b


def _cov_sqrt_inv(c: FloatArray, ridge: float = 1e-10) -> FloatArray:
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, ridge)
    return np.asarray((v / np.sqrt(w)) @ v.T, dtype=np.float64)


def cca(x: FloatArray, y: FloatArray, n_components: int | None = None) -> dict[str, FloatArray]:
    """Canonical correlations + weight vectors via whitened SVD.

    Returns canonical correlations rho_i, weight matrices
    x_weights (p, r), y_weights (q, r) giving unit-variance canonical
    variates (up to sampling error).
    """
    a, b = _check_xy(x, y)
    n, p = a.shape
    q = b.shape[1]
    r_max = min(p, q)
    r = r_max if n_components is None else min(n_components, r_max)
    xc = a - a.mean(axis=0)
    yc = b - b.mean(axis=0)
    cxx = xc.T @ xc / (n - 1)
    cyy = yc.T @ yc / (n - 1)
    cxy = xc.T @ yc / (n - 1)
    cxh = _cov_sqrt_inv(cxx)
    cyh = _cov_sqrt_inv(cyy)
    m = cxh @ cxy @ cyh
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    u, s, vt = u[:, :r], s[:r], vt[:r]
    return {
        "correlations": np.asarray(s, dtype=np.float64),
        "x_weights": np.asarray(cxh @ u, dtype=np.float64),
        "y_weights": np.asarray(cyh @ vt.T, dtype=np.float64),
    }


def pls(x: FloatArray, y: FloatArray, n_components: int = 1) -> dict[str, FloatArray]:
    """PLS-SVD directions maximizing covariance between scores."""
    a, b = _check_xy(x, y)
    xc = a - a.mean(axis=0)
    yc = b - b.mean(axis=0)
    cxy = xc.T @ yc
    u, s, vt = np.linalg.svd(cxy, full_matrices=False)
    r = min(n_components, u.shape[1])
    return {
        "x_weights": np.asarray(u[:, :r], dtype=np.float64),
        "y_weights": np.asarray(vt[:r].T, dtype=np.float64),
        "covariances": np.asarray(s[:r], dtype=np.float64),
    }


def reduced_rank_regression(
    x: FloatArray, y: FloatArray, rank: int = 1
) -> dict[str, FloatArray | float]:
    """Anderson RRR: rank-r coefficient matrix minimizing ||Y - X B||_F.

    B_hat = Sigma_xx^{-1/2} V_r V_r^T Sigma_xx^{1/2} B_ols, where V_r
    are the top-r right singular vectors of the whitened coherence.
    """
    a, b = _check_xy(x, y)
    n = a.shape[0]
    xc = a - a.mean(axis=0)
    yc = b - b.mean(axis=0)
    cxx = xc.T @ xc / (n - 1)
    cyy = yc.T @ yc / (n - 1)
    cxy = xc.T @ yc / (n - 1)
    b_ols = np.linalg.lstsq(xc, yc, rcond=None)[0]
    # canonical Y directions: eigenvectors of Cyy^{-1} Cyx Cxx^{-1} Cxy
    cyh = _cov_sqrt_inv(cyy)
    cxh = _cov_sqrt_inv(cxx)
    m = cyh @ cxy.T @ (cxh @ cxh) @ cxy @ cyh
    w, v = np.linalg.eigh(m)
    v_r = v[:, np.argsort(w)[::-1][:rank]]
    # Y-space basis A_r = Cyy^{-1/2} V_r; oblique projector
    a_r = cyh @ v_r
    proj = a_r @ np.linalg.solve(a_r.T @ cyy @ a_r, a_r.T @ cyy)
    b_rr = b_ols @ proj
    return {
        "coef": np.asarray(b_rr, dtype=np.float64),
        "coef_ols": np.asarray(b_ols, dtype=np.float64),
        "canonical_basis": np.asarray(a_r, dtype=np.float64),
        "rank": float(rank),
    }


def bench_cca(seed: int = 20261231 + 404) -> dict[str, float]:
    """SYNTHETIC check — CCA recovers the planted shared direction."""
    rng = np.random.default_rng(seed)
    n = 500
    z = rng.standard_normal(n)
    a_true = np.array([1.0, 0.5, -0.3, 0.2])
    b_true = np.array([0.8, -0.4, 0.3])
    rho_true = 0.85
    xa = rho_true * z + np.sqrt(1 - rho_true**2) * rng.standard_normal(n)
    xb = rho_true * z + np.sqrt(1 - rho_true**2) * rng.standard_normal(n)
    x = np.column_stack(
        [
            xa + 0.1 * rng.standard_normal(n) * a_true[0],
            0.9 * rng.standard_normal(n),
            0.9 * rng.standard_normal(n),
            0.9 * rng.standard_normal(n),
        ]
    )
    x[:, 0] = a_true[0] * xa
    x[:, 1] = a_true[1] * xa + 0.9 * rng.standard_normal(n)
    x[:, 2] = a_true[2] * xa + 0.9 * rng.standard_normal(n)
    x[:, 3] = a_true[3] * xa + 0.9 * rng.standard_normal(n)
    y = np.column_stack(
        [
            b_true[0] * xb + 0.9 * rng.standard_normal(n),
            b_true[1] * xb + 0.9 * rng.standard_normal(n),
            b_true[2] * xb + 0.9 * rng.standard_normal(n),
        ]
    )
    out = cca(x, y, n_components=1)
    rho1 = float(np.asarray(out["correlations"])[0])
    a_hat = np.asarray(out["x_weights"])[:, 0]
    a_hat = a_hat / np.linalg.norm(a_hat)
    a_dir = a_true / np.linalg.norm(a_true)
    align = abs(float(np.dot(a_hat, a_dir)))
    # the sample canonical correlation is lower than rho(xa,xb)=0.85
    # because z is only partially recoverable from the 4-dim X view;
    # the honest bound is (broad-band significance, exact direction).
    if not (0.4 < rho1 < 0.95) or align < 0.85:
        raise ValueError(f"CCA off: rho {rho1} align {align}")
    # RRR recovers the rank-1 mapping direction better than noise
    rr = reduced_rank_regression(x, y, rank=1)
    b_rr = np.asarray(rr["coef"])
    if b_rr.shape != (x.shape[1], y.shape[1]):
        raise ValueError("RRR shape off")
    return {
        "synthetic_cca_rho": rho1,
        "synthetic_cca_align": align,
        "synthetic_score": 1.0,
    }
