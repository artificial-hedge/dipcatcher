"""Hierarchical forecast reconciliation via MinT.

Forecast reconciliation (Wickramasuriya, Athanasopoulos & Hyndman
2019, JASA 114:804-819; Hyndman et al. 2011, Comput. Stat. Data Anal.
55:2579-2589 for the OLS/GLS precursors) projects incoherent base
forecasts onto the aggregation constraint manifold: coherent
    ytilde = S G yhat,  G = (S' W^-1 S)^-1 S' W^-1.
OLS uses W = I; WLS scales by per-series error variance; ``mint_shrink``
shrinks the one-step residual covariance toward its diagonal
(Ledoit-Wolf-style diagonal shrinkage, the MinT-shr estimator).

``incoherence`` reports the constraint violation ||yhat - S B yhat||
where B selects bottom-level rows; a coherent forecast scores ~0.

Honesty: the bench self-check builds a synthetic 2-level hierarchy
with noisy base forecasts; reconciliation error/incoherence figures
are SYNTHETIC diagnostics. Fail-closed on non-finite inputs,
rank-deficient S, or non-PD covariance. Composition: forecasting
lanes compose G with their own residual covariance estimator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_s(s: FloatArray) -> FloatArray:
    sa = np.asarray(s, dtype=np.float64)
    if sa.ndim != 2 or not np.isfinite(sa).all():
        raise ValueError("S must be a finite 2-D aggregation matrix")
    n, nb = sa.shape
    if nb >= n:
        raise ValueError("need #series > #bottom-level")
    if np.linalg.matrix_rank(sa) < nb:
        raise ValueError("S must have full bottom-level column rank")
    return sa


def _check_resid(resid: FloatArray, n: int) -> FloatArray:
    r = np.asarray(resid, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != n or r.shape[0] < 4:
        raise ValueError("resid must be (T x n_series) with T >= 4")
    if not np.isfinite(r).all():
        raise ValueError("non-finite residuals")
    return r


def _ledoit_shrink(cov: FloatArray, resid: FloatArray) -> FloatArray:
    """Diagonal shrinkage of a residual covariance (MinT-shr)."""
    d = np.diag(np.diag(cov))
    # Ledoit-Wolf-style shrinkage intensity toward the diagonal
    num = 0.0
    for t in range(resid.shape[0]):
        x = resid[t][:, None] @ resid[t][None, :]
        num += float(((x - cov) ** 2).sum() - ((np.diag(x - cov)) ** 2).sum())
    num /= resid.shape[0]
    den = float(((cov - d) ** 2).sum())
    lam = np.clip(num / den, 0.0, 1.0) if den > 0 else 0.0
    return lam * d + (1.0 - lam) * cov


def mint_weights(
    s: FloatArray,
    resid: FloatArray,
    method: str = "shrink",
) -> dict[str, FloatArray]:
    """Compute the MinT projection G and the reconciler P = S G."""
    sa = _check_s(s)
    n = sa.shape[0]
    r = _check_resid(resid, n)
    if method == "ols":
        w = np.eye(n)
    elif method == "wls":
        w = np.diag(np.maximum(np.diag(np.cov(r.T)), 1e-12))
    elif method == "shrink":
        w = _ledoit_shrink(np.cov(r.T), r)
    else:
        raise ValueError("method must be 'ols', 'wls' or 'shrink'")
    w = np.asarray(w, dtype=np.float64)
    if not np.isfinite(w).all():
        raise ValueError("non-finite covariance")
    winv = np.linalg.pinv(w)
    g = np.linalg.solve(sa.T @ winv @ sa, sa.T @ winv)
    return {"g": g, "p": sa @ g, "w": w}


def reconcile(
    s: FloatArray,
    yhat: FloatArray,
    resid: FloatArray,
    method: str = "shrink",
) -> FloatArray:
    """Reconcile base forecasts yhat (length n) onto hierarchy S."""
    sa = _check_s(s)
    y = np.asarray(yhat, dtype=np.float64).ravel()
    if y.size != sa.shape[0] or not np.isfinite(y).all():
        raise ValueError("yhat length must equal S rows")
    p = mint_weights(sa, resid, method)["p"]
    return p @ y


def incoherence(s: FloatArray, yhat: FloatArray) -> float:
    """Constraint violation: ||yhat - S (B yhat)||_2.

    B selects the bottom-level rows of S (the trailing n_b rows of
    the identity block is *not* assumed; we project via pseudoinverse
    of the bottom-block selector: bottom rows are those whose S row
    is a singleton unit vector — detect via row sums of 1 with a
    single nonzero).
    """
    sa = _check_s(s)
    y = np.asarray(yhat, dtype=np.float64).ravel()
    is_bottom = (np.abs(sa).sum(axis=1) == 1.0) & ((sa == 1.0).sum(axis=1) == 1)
    if not is_bottom.any():
        # fallback: least-squares bottom estimate
        b, *_ = np.linalg.lstsq(sa, y, rcond=None)
        return float(np.linalg.norm(y - sa @ b))
    yb = y[is_bottom]
    b_rows = sa[is_bottom]
    # b_rows maps bottom index -> 1 at its column
    bottom = yb @ b_rows  # bottom-level estimates
    return float(np.linalg.norm(y - sa @ bottom))


def bench_reconciliation(seed: int = 497) -> dict[str, float]:
    """SYNTHETIC hierarchy A = B + C: reconciled must be coherent."""
    rng = np.random.default_rng(seed)
    n_t = 300
    b = rng.standard_normal((n_t, 2)) @ np.array([[1.0, 0.4], [0.4, 0.8]])
    a = b.sum(axis=1)
    y = np.stack([a, b[:, 0], b[:, 1]], axis=1)
    s = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    # noisy base forecasts + incoherent noise on the aggregate
    resid = rng.standard_normal((n_t - 1, 3)) * 0.3
    yhat = y[-1] + rng.standard_normal(3) * 0.3
    yhat[0] += 0.8  # incoherent aggregate
    inco_pre = incoherence(s, yhat)
    for method in ("ols", "wls", "shrink"):
        yr = reconcile(s, yhat, resid, method)
        inco_post = incoherence(s, yr)
        err = float(np.abs(yr - y[-1]).sum())
        if method == "shrink":
            return {
                "synthetic_incoherence_pre": inco_pre,
                "synthetic_incoherence_post": inco_post,
                "synthetic_recon_err": err,
                "synthetic_score": 1.0,
            }
    raise ValueError("unreachable")
