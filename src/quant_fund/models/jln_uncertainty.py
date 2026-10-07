"""Jurado-Ludvigson-Ng (2015) aggregate uncertainty factor.

JLN define macro uncertainty as the common factor of individual
series' conditional h-step-ahead forecast-error variances:

    U_i(t) = sqrt( E[ (y_{i,t+h} - F_t y_{i,t+h})^2 | F_t ] )

Operational version here: fit an AR(1) to each series, form h-step
forecast-error variance sigma_i^2(h) = sig_i^2 * sum_{j<h} phi^{2j},
then take the first principal component of the standardized
series-level uncertainty paths (rolling conditional vol of the
one-step forecast errors). The loadings/sign are normalized so the
index rises when the economy is uncertain.

References
----------
- Jurado, Ludvigson & Ng (2015) AER 105, "Measuring uncertainty".
- Ludvigson, Ma & Ng (2021) AER, "Uncertainty and business cycles".

Honesty
-------
This is an index, not a probability; partial identification is honest
— the bench gates on correlation with the planted latent uncertainty
factor, not on level recovery. SYNTHETIC only.

Composition
-----------
Called by ``quant_fund.research.benches_w63.bench_jln``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ar1(y: FloatArray) -> tuple[float, float]:
    """OLS AR(1): returns (phi, sigma_eps)."""
    x0, x1 = y[:-1], y[1:]
    xm = x0.mean()
    den = float(np.sum((x0 - xm) ** 2))
    phi = float(np.sum((x0 - xm) * (x1 - x1.mean())) / den) if den > 0 else 0.0
    phi = float(np.clip(phi, -0.98, 0.98))
    resid = x1 - x1.mean() - phi * (x0 - xm)
    return phi, float(np.sqrt(np.mean(resid**2)))


def jln_uncertainty(panel: FloatArray, horizon: int = 3, window: int = 24) -> dict[str, FloatArray]:
    """Common uncertainty factor across a panel.

    Parameters
    ----------
    panel:
        (n_series, n_time) matrix of stationary series.
    horizon:
        Forecast horizon used for the per-series uncertainty weights.
    window:
        Rolling window for conditional forecast-error vol.

    Returns
    -------
    dict with ``index`` (n_time,), ``series_u`` (n_series, n_time)
    rolling uncertainties, and ``share_pc1`` the variance share of the
    first principal component.
    """
    p = np.asarray(panel, dtype=float)
    if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] < window + horizon + 5:
        raise ValueError("panel must be (n_series>=3, long enough)")
    if not np.all(np.isfinite(p)):
        raise ValueError("panel must be finite")
    if not (horizon >= 1 and 5 <= window < p.shape[1] - horizon):
        raise ValueError("bad horizon/window")
    n_s, n_t = p.shape
    u = np.full((n_s, n_t), np.nan)
    for i in range(n_s):
        phi, sig = _ar1(p[i])
        resid = p[i, 1:] - np.mean(p[i, 1:]) - phi * (p[i, :-1] - np.mean(p[i, :-1]))
        w_h = float(np.sqrt(sum(phi ** (2 * j) for j in range(horizon))))
        vol = np.empty(n_t)
        for t in range(n_t):
            lo = max(0, t - window)
            seg = resid[max(0, lo - 1) : t] if lo else resid[:t]
            vol[t] = np.std(seg) if seg.size > 2 else np.nan
        u[i] = w_h * vol
    # Standardize each series' uncertainty path, first PC.
    z = (u - np.nanmean(u, axis=1, keepdims=True)) / np.nanstd(u, axis=1, keepdims=True)
    valid = np.all(np.isfinite(z), axis=0)
    zv = z[:, valid]
    cov = np.cov(zv)
    w, v = np.linalg.eigh(cov)
    pc = v[:, -1]
    idx = pc @ zv
    if np.corrcoef(idx, np.nanmean(u[:, valid], axis=0))[0, 1] < 0:
        idx = -idx
    out = np.full(n_t, np.nan)
    out[valid] = idx
    share = float(w[-1] / np.sum(np.maximum(w, 0)))
    return {"index": out, "series_u": u, "share_pc1": np.array([share])}


def bench_jln(seed: int = 20261231 + 370) -> dict[str, float]:
    """SYNTHETIC check — recover a planted latent uncertainty factor."""
    rng = np.random.default_rng(seed)
    n_s, n_t = 10, 320
    # Latent stochastic-vol uncertainty factor (slow-moving, >0).
    u_t = np.exp(np.cumsum(rng.standard_normal(n_t) * 0.04))
    u_t = u_t / np.mean(u_t)
    panel = np.empty((n_s, n_t))
    for i in range(n_s):
        phi = rng.uniform(0.2, 0.6)
        e = np.empty(n_t)
        e[0] = 0.0
        for t in range(1, n_t):
            e[t] = phi * e[t - 1] + np.sqrt(u_t[t]) * rng.standard_normal() * 0.8
        panel[i] = e + rng.standard_normal(n_t) * 0.05
    res = jln_uncertainty(panel, horizon=3, window=30)
    idx = res["index"]
    valid = np.isfinite(idx)
    cor = float(np.corrcoef(idx[valid], np.log(u_t)[valid])[0, 1])
    share = float(res["share_pc1"][0])
    if cor < 0.55:
        raise ValueError("uncertainty index fails to track the latent factor")
    if share < 0.35:
        raise ValueError("first PC does not dominate — weak commonality")
    return {
        "synthetic_jln_corr": cor,
        "synthetic_jln_share_pc1": share,
        "synthetic_jln_mean_u": float(np.nanmean(res["series_u"])),
        "synthetic_score": 1.0,
    }
