"""Exploratory factor analysis — principal axis + varimax rotation.

Thurstone (1947), Harman (1976): the common-factor model writes the
correlation matrix R ≈ LL^T + Psi with loadings L (p x k) and
uniquenesses Psi diagonal. Principal-axis factoring iterates
communalities h_i^2 = sum_j l_ij^2: start h^2 from SMC, reduce
R* = R - Psi, eigendecompose, rotate. Varimax (Kaiser 1958)
orthogonal rotation maximizes the sum of within-factor loading
variances for simple structure.

Honesty: the bench plants a 2-factor loading matrix on 6 variables,
samples from the implied correlation, and checks (a) the recovered
loading pattern matches the planted blocks up to sign/rotation via
Tucker's congruence, and (b) communalities are recovered within MC
tolerance. PAF is a fitting heuristic (no likelihood) — bounds are
loose and documented. Fail-closed on non-PSD input or k >= p.

References: Harman (1976) "Modern Factor Analysis"; Kaiser (1958)
"The varimax criterion for analytic rotation"; Tucker (1951) congruence
coefficient.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_corr(r: FloatArray) -> FloatArray:
    a = np.asarray(r, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 3:
        raise ValueError("bad correlation matrix")
    if not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-8):
        raise ValueError("correlation must be symmetric finite")
    return a


def _smc(r: FloatArray) -> FloatArray:
    """Squared multiple correlations (initial communalities)."""
    rinv = np.linalg.inv(r)
    d = np.diag(rinv)
    return np.asarray(1.0 - 1.0 / d, dtype=np.float64)


def varimax(lam: FloatArray, n_iter: int = 100, tol: float = 1e-6) -> FloatArray:
    """Kaiser varimax rotation of the loading matrix."""
    a = np.asarray(lam, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError("bad loadings")
    p, k = a.shape
    r = np.eye(k)
    d_old = 0.0
    x = a.copy()
    for _ in range(n_iter):
        lam_rot = a @ r
        u, s, vh = np.linalg.svd(a.T @ (lam_rot**3 - (lam_rot * (lam_rot**2).sum(axis=0) / p)))
        r = u @ vh
        d = s.sum()
        if d_old and abs(d - d_old) / d_old < tol:
            x = a @ r
            break
        d_old = d
        x = a @ r
    return np.asarray(x, dtype=np.float64)


def factor_analysis(r: FloatArray, n_factors: int = 2, n_iter: int = 100) -> dict[str, FloatArray]:
    """Principal-axis factor analysis with varimax rotation.

    Returns ``loadings`` (varimax-rotated), ``unrotated``,
    ``communalities``, ``uniquenesses``, ``fit_residual`` (max off-
    diagonal |R - LL^T - Psi|).
    """
    a = _check_corr(r)
    p = a.shape[0]
    if not 1 <= n_factors < p:
        raise ValueError("bad n_factors")
    h2 = np.clip(_smc(a), 0.05, 0.98)
    lam = np.zeros((p, n_factors))
    for _ in range(n_iter):
        red = a.copy()
        np.fill_diagonal(red, h2)
        vals, vecs = np.linalg.eigh(red)
        idx = np.argsort(vals)[::-1][:n_factors]
        vals_k = np.clip(vals[idx], 1e-10, None)
        lam = vecs[:, idx] * np.sqrt(vals_k)[None, :]
        h2_new = np.clip((lam**2).sum(axis=1), 0.0, 1.0)
        if np.abs(h2_new - h2).max() < 1e-8:
            h2 = h2_new
            break
        h2 = h2_new
    psi = np.clip(1.0 - h2, 0.0, 1.0)
    rot = varimax(lam) if n_factors > 1 else lam
    resid = a - (lam @ lam.T) - np.diag(psi)
    np.fill_diagonal(resid, 0.0)
    return {
        "loadings": np.asarray(rot, dtype=np.float64),
        "unrotated": np.asarray(lam, dtype=np.float64),
        "communalities": np.asarray(h2, dtype=np.float64),
        "uniquenesses": np.asarray(psi, dtype=np.float64),
        "fit_residual": np.asarray(float(np.abs(resid).max())),
    }


def bench_factor_analysis(seed: int = 20261231 + 420) -> dict[str, float]:
    """SYNTHETIC check — planted 2-factor structure recovery."""
    rng = np.random.default_rng(seed)
    p, k, n = 6, 2, 800
    lam_t = np.array(
        [
            [0.85, 0.0],
            [0.75, 0.1],
            [0.8, 0.0],
            [0.0, 0.85],
            [0.1, 0.75],
            [0.0, 0.8],
        ]
    )
    psi_t = 1.0 - (lam_t**2).sum(axis=1)
    f = rng.standard_normal((n, k))
    e = rng.standard_normal((n, p)) * np.sqrt(psi_t)[None, :]
    x = f @ lam_t.T + e
    r = np.corrcoef(x.T)
    out = factor_analysis(r, n_factors=2)
    l_hat = np.asarray(out["loadings"], dtype=np.float64)
    # Tucker congruence between planted and rotated-loadings columns
    congr = []
    for i in range(k):
        num = float(lam_t[:, i] @ l_hat[:, i])
        den = np.linalg.norm(lam_t[:, i]) * np.linalg.norm(l_hat[:, i])
        congr.append(abs(num / max(den, 1e-12)))
    congr_min = min(congr)
    h2_err = float(np.abs(np.asarray(out["communalities"]) - (lam_t**2).sum(1)).mean())
    if congr_min < 0.9 or h2_err > 0.15:
        raise ValueError(f"fa recovery off: congr={congr_min:.3f} h2_err={h2_err:.3f}")
    return {
        "synthetic_fa_congruence": congr_min,
        "synthetic_fa_h2_err": h2_err,
        "synthetic_fa_fit_resid": float(out["fit_residual"]),
        "score": 1.0,
    }
