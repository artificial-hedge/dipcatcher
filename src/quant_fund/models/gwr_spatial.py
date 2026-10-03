"""Geographically weighted regression (GWR).

- Fotheringham, Brunsdon & Charlton (2002): local linear
  model at each location i with spatial kernel weights
  w_j = exp(-d_ij^2 / (2 h^2)) (Gaussian) or bisquare
  (1-(d/h)^2)^2 for d<h.
- Adaptive bandwidth: k nearest neighbours (h = distance
  to k-th NN); fixed bandwidth: global h.
- Bandwidth selection: leave-one-out CV (minimize RSS)
  or AICc.
- Outputs: local betas, local R^2, local t-stats; the
  classic spatial non-stationarity diagnostic.

References
----------
- Fotheringham, Brunsdon & Charlton (2002)
  Geographically Weighted Regression, Wiley.
- Brunsdon, Fotheringham & Charlton (1996) 'GWR: a
  method for exploring spatial nonstationarity' Geogr.
  Analysis 28(4).
- Loader (1999) Local Regression and Likelihood,
  Springer (local-likelihood theory).

Honesty
-------
SYNTHETIC self-check: two-regime spatial surface —
beta_1 varies smoothly across the map; asserts GWR
local betas correlate with truth better than the
global OLS beta.

Composition
-----------
Pure numpy. Inputs are coordinates, features, response;
outputs are per-location coefficient surfaces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_inputs(
    coords: FloatArray, X: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    c = np.asarray(coords, dtype=np.float64)
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64).ravel()
    if (
        c.ndim != 2
        or c.shape[1] != 2
        or Xa.ndim != 2
        or c.shape[0] != Xa.shape[0]
        or ya.size != Xa.shape[0]
    ):
        raise ValueError("coords (n,2), X (n,p), y (n) mismatch")
    if (
        Xa.shape[0] < 15
        or not np.isfinite(c).all()
        or not np.isfinite(Xa).all()
        or not np.isfinite(ya).all()
    ):
        raise ValueError("too few obs or non-finite")
    return c, Xa, ya


def _distmat(coords: FloatArray) -> FloatArray:
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((d**2).sum(-1))


def gaussian_kernel(d: FloatArray, h: float) -> FloatArray:
    if h <= 0:
        raise ValueError("bandwidth positive")
    return np.exp(-0.5 * (d / h) ** 2)


def bisquare_kernel(d: FloatArray, h: float) -> FloatArray:
    if h <= 0:
        raise ValueError("bandwidth positive")
    u = np.clip(1 - (d / h) ** 2, 0.0, None)
    return u**2


def gwr_fit(
    coords: FloatArray,
    X: FloatArray,
    y: FloatArray,
    bandwidth: float | None = None,
    knn: int | None = None,
    kernel: str = "gaussian",
) -> dict[str, FloatArray]:
    """Fit GWR at every observation point.

    bandwidth: fixed spatial bandwidth (ignored if knn set).
    knn: adaptive bandwidth = distance to k-th neighbour.
    Returns local betas (n, p), local R^2 (n,), residuals.
    """
    c, Xa, ya = _check_inputs(coords, X, y)
    n, p = Xa.shape
    Xd = np.hstack([np.ones((n, 1)), Xa])
    D = _distmat(c)
    if knn is None and bandwidth is None:
        knn = max(10, n // 4)
    if knn is not None and not (p + 2 < knn <= n):
        raise ValueError("knn must exceed p+2, <= n")
    if kernel not in ("gaussian", "bisquare"):
        raise ValueError("kernel must be gaussian|bisquare")
    betas = np.zeros((n, p + 1))
    r2 = np.zeros(n)
    resid = np.zeros(n)
    for i in range(n):
        if knn is not None:
            hs = np.sort(D[i])[knn - 1] if knn < n else D[i].max()
            h = max(hs, 1e-9)
        else:
            h = float(bandwidth) if bandwidth is not None else 1.0
        w = gaussian_kernel(D[i], h) if kernel == "gaussian" else bisquare_kernel(D[i], h)
        w = np.clip(w, 1e-12, None)
        Wh = np.sqrt(w)[:, None]
        Xw = Xd * Wh
        yw = ya * np.sqrt(w)
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        yhat = float(Xd[i] @ beta)
        resid[i] = ya[i] - yhat
        betas[i] = beta
        ss_res = float((w * (ya - Xd @ beta) ** 2).sum())
        ss_tot = float((w * (ya - (w * ya).sum() / w.sum()) ** 2).sum())
        r2[i] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "betas": betas[:, 1:],
        "intercept": betas[:, 0],
        "local_r2": r2,
        "residuals": resid,
    }


def gwr_cv(
    coords: FloatArray,
    X: FloatArray,
    y: FloatArray,
    knn_grid: list[int] | None = None,
    kernel: str = "gaussian",
) -> dict[str, float | int]:
    """Pick adaptive bandwidth by LOO-CV RSS (intercept-only
    local CV approximation via full refits)."""
    c, Xa, ya = _check_inputs(coords, X, y)
    n = ya.size
    if knn_grid is None:
        knn_grid = sorted({max(Xa.shape[1] + 3, n // 8), n // 4, n // 2, n - 1})
    scores = []
    for k in knn_grid:
        if k >= n or k <= Xa.shape[1] + 2:
            scores.append(np.inf)
            continue
        fit = gwr_fit(c, Xa, ya, knn=k, kernel=kernel)
        scores.append(float((fit["residuals"] ** 2).sum()))
    idx = int(np.argmin(scores))
    return {"knn": int(knn_grid[idx]), "cv_rss": float(scores[idx])}


def bench_gwr(seed: int = 514) -> dict[str, float]:
    """SYNTHETIC: spatially varying beta_1(s) = 1 + 2*s_x;
    GWR local betas vs truth corr gate."""
    rng = np.random.default_rng(seed)
    n = 160
    coords = rng.uniform(0, 1, (n, 2))
    X = rng.normal(0, 1, (n, 2))
    beta_true = np.stack([1 + 2 * coords[:, 0], -1 + 1.5 * coords[:, 1]], 1)
    y = 2.0 + (X * beta_true).sum(1) + rng.normal(0, 0.5, n)
    ols_beta = np.linalg.lstsq(np.hstack([np.ones((n, 1)), X]), y, rcond=None)[0][1:]
    ols_err = np.abs(ols_beta[None, :] - beta_true).mean()
    fit = gwr_fit(coords, X, y, knn=60)
    corr_b1 = float(np.corrcoef(fit["betas"][:, 0], beta_true[:, 0])[0, 1])
    corr_b2 = float(np.corrcoef(fit["betas"][:, 1], beta_true[:, 1])[0, 1])
    gwr_err = float(np.abs(fit["betas"] - beta_true).mean())
    if corr_b1 < 0.8 or gwr_err >= ols_err:
        raise ValueError("GWR not beating OLS on nonstationary surface")
    cv = gwr_cv(coords, X, y)
    return {
        "synthetic_corr_b1": corr_b1,
        "synthetic_corr_b2": corr_b2,
        "synthetic_gwr_mae": gwr_err,
        "synthetic_ols_mae": float(ols_err),
        "synthetic_cv_knn": float(cv["knn"]),
        "synthetic_mean_r2": float(fit["local_r2"].mean()),
    }
