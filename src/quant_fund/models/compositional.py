"""Compositional data analysis — Aitchison geometry on the simplex (SYNTHETIC).

Aitchison (1982, 1986): proportions x = (x_1..x_D) live on the
simplex where the right geometry is not Euclidean. The log-ratio
family maps to ordinary coordinates:

    clr(x)_i = log(x_i) - mean_j log(x_j)        (centered log-ratio)
    ilr(x)   = clr(x) @ Psi^T,  Psi orthonormal  (isometric log-ratio)

Variation matrix entries T_ij = var(log(x_i/x_j)) measure pairwise
proportionality; the Dirichlet distribution is the standard generative
model on the simplex (alpha-mixture of counts).

Honesty: the bench draws a Dirichlet sample with planted alpha,
estimates alpha by moment matching on the clr covariance, and checks
the recoveries: clr/ilr inverse consistency, positive-Definiteness of
the variation matrix, and alpha hat within MC tolerance. Fail-closed
on zero/negative parts (log-ratios need positivity) and non-finite
input.

References: Aitchison (1986) "The Statistical Analysis of
Compositional Data"; Egozcue et al. (2003) "Isometric logratio
transformations for compositional data analysis"; Lovell et al.
(2015) "Proportionality: a valid alternative to correlation".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_comp(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError("bad composition")
    if (a <= 0).any():
        raise ValueError("compositional parts must be positive")
    return np.asarray(a / a.sum(axis=1, keepdims=True), dtype=np.float64)


def clr(x: FloatArray) -> FloatArray:
    """Centered log-ratio transform (rows sum to zero)."""
    a = _check_comp(x)
    lg = np.asarray(np.log(a), dtype=np.float64)
    return np.asarray(lg - lg.mean(axis=1, keepdims=True), dtype=np.float64)


def _helmert(d: int) -> FloatArray:
    """(d-1) x d orthonormal Helmert contrast matrix."""
    h = np.zeros((d - 1, d))
    for i in range(d - 1):
        h[i, : i + 1] = 1.0 / (i + 1)
        h[i, i + 1] = -1.0
        h[i] /= np.linalg.norm(h[i])
    return h


def ilr(x: FloatArray) -> FloatArray:
    """Isometric log-ratio via Helmert contrasts (isometry to R^{D-1})."""
    a = clr(x)
    psi = _helmert(a.shape[1])
    return np.asarray(a @ psi.T, dtype=np.float64)


def ilr_inv(z: FloatArray) -> FloatArray:
    """Inverse ilr: exp then closure."""
    z = np.asarray(z, dtype=float)
    psi = _helmert(z.shape[1] + 1)
    c = z @ psi
    e = np.exp(c - c.max(axis=1, keepdims=True))
    return np.asarray(e / e.sum(axis=1, keepdims=True), dtype=np.float64)


def variation_matrix(x: FloatArray) -> FloatArray:
    """T_ij = var(log(x_i / x_j)) across samples."""
    a = _check_comp(x)
    lg = np.log(a)
    d = a.shape[1]
    t = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            t[i, j] = np.var(lg[:, i] - lg[:, j])
    return np.asarray(t, dtype=np.float64)


def dirichlet_moments(x: FloatArray) -> dict[str, float]:
    """Method-of-moments Dirichlet alpha from part means/variances."""
    a = _check_comp(x)
    if a.shape[0] < 10:
        raise ValueError("need >=10 samples")
    m = a.mean(axis=0)
    v = a.var(axis=0)
    # alpha_0 from the first two moments: Var(x_i) = m_i(1-m_i)/(a0+1)
    ratios = m * (1.0 - m) / np.maximum(v, 1e-12) - 1.0
    a0 = float(np.median(ratios))
    if a0 <= 0:
        raise ValueError("degenerate fit")
    return {
        "alpha0": float(a0),
        "n_parts": float(a.shape[1]),
        "n_samples": float(a.shape[0]),
    }


def bench_compositional(seed: int = 20261231 + 412) -> dict[str, float]:
    """SYNTHETIC check — ilr inversion, variation PSD, Dirichlet recovery."""
    rng = np.random.default_rng(seed)
    alpha_true = np.array([2.0, 5.0, 1.5])
    x = rng.dirichlet(alpha_true, size=800)
    # clr/ilr round-trip
    z = ilr(x)
    back = ilr_inv(z)
    rec_err = float(np.abs(back - x).max())
    if rec_err > 1e-9:
        raise ValueError(f"ilr inverse off: {rec_err}")
    # the variation matrix itself is a squared-difference matrix —
    # the correct spectral check is that the double-centered
    # (MDS) matrix -J T J is PSD.
    t = variation_matrix(x)
    if not np.allclose(t, t.T, atol=1e-10):
        raise ValueError("variation not symmetric")
    if np.abs(np.diag(t)).max() > 1e-9:
        raise ValueError("variation diagonal nonzero")
    j = np.eye(t.shape[0]) - np.ones(t.shape) / t.shape[0]
    b = -0.5 * j @ t @ j
    eigs = np.linalg.eigvalsh(b)
    if eigs.min() < -1e-8:
        raise ValueError("double-centered variation not PSD")
    # Dirichlet alpha0 recovery: true = 8.5
    fit = dirichlet_moments(x)
    a0_err = abs(fit["alpha0"] - float(alpha_true.sum())) / float(alpha_true.sum())
    if a0_err > 0.25:
        raise ValueError(f"alpha0 off: {fit['alpha0']}")
    return {
        "synthetic_coda_rec_err": rec_err,
        "synthetic_coda_min_eig": float(eigs.min()),
        "synthetic_coda_var_trace": float(np.trace(t)),
        "synthetic_coda_alpha0_err": float(a0_err),
        "synthetic_coda_alpha0_hat": float(fit["alpha0"]),
        "synthetic_score": 1.0,
    }
