"""HAR realized-volatility models (Corsi 2009) and HARQ extension (SYNTHETIC)
(Bollerslev, Patton & Quaedvlieg 2016).

HAR-RV: RV_{t+1} = b0 + b_d RV_t + b_w mean(RV_{t-4..t})
       + b_m mean(RV_{t-21..t}) + e  (daily/weekly/monthly cascade).
HARQ: the daily coefficient becomes b_d + b_q * sqrt(RQ_t), where RQ
is realized quarticity -- heterogeneous persistence by measurement
accuracy.

OLS with Newey-West standard errors. Fail-closed: non-positive RV,
non-finite inputs, insufficient history.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check(rv: Array, rq: Array | None = None) -> tuple[Array, Array | None]:
    rvv = np.asarray(rv, dtype=float).ravel()
    if rvv.size < 60 or not np.isfinite(rvv).all() or (rvv <= 0).any():
        raise ValueError("rv must be finite, positive, >= 60 obs")
    if rq is not None:
        rqq = np.asarray(rq, dtype=float).ravel()
        if rqq.size != rvv.size or not np.isfinite(rqq).all() or (rqq < 0).any():
            raise ValueError("rq must be finite, nonnegative, aligned")
        return rvv, rqq
    return rvv, None


def _nw_se(x: Array, resid: Array) -> Array:
    """Newey-West SEs for OLS coefficients."""
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    meat = np.zeros((k, k))
    for t in range(n):
        xt = x[t : t + 1].T
        meat += resid[t] ** 2 * (xt @ xt.T)
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        for t in range(lag, n):
            g = (
                resid[t]
                * resid[t - lag]
                * (np.outer(x[t], x[t - lag]) + np.outer(x[t - lag], x[t]))
            )
            meat += w * g
    cov = xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def _har_design(rv: Array, lags: tuple[int, int, int]) -> Array:
    n = rv.size
    d, w, m = lags
    x = np.full((n, 4), np.nan)
    x[:, 0] = 1.0
    for t in range(m, n):
        x[t, 1] = rv[t - 1]  # daily
        x[t, 2] = rv[max(0, t - w) : t].mean()  # weekly
        x[t, 3] = rv[t - m : t].mean()  # monthly
    return x


def har_rv_fit(rv: Array, lags: tuple[int, int, int] = (1, 5, 22)) -> dict[str, Array | float]:
    """HAR-RV OLS fit. Returns coefs (b0, bd, bw, bm), NW SEs, fitted, resid."""
    rvv, _ = _check(rv)
    d, w, m = lags
    if not (0 < d < w < m):
        raise ValueError("lags must satisfy 0 < daily < weekly < monthly")
    if rvv.size < m + 30:
        raise ValueError("insufficient history for the monthly lag")
    x = _har_design(rvv, lags)
    valid = np.isfinite(x[:, 1])
    xv, yv = x[valid], rvv[valid]
    coef = np.linalg.lstsq(xv, yv, rcond=None)[0]
    fitted = xv @ coef
    resid = yv - fitted
    se = _nw_se(xv, resid)
    return {
        "coef": coef,
        "se": se,
        "fitted": fitted,
        "resid": resid,
        "r2": 1.0 - float(resid @ resid) / float(((yv - yv.mean()) ** 2).sum()),
    }


def harq_fit(
    rv: Array, rq: Array, lags: tuple[int, int, int] = (1, 5, 22)
) -> dict[str, Array | float]:
    """HARQ: daily term = (b_d + b_q sqrt(RQ_t)) * RV_t."""
    rvv, rqq = _check(rv, rq)
    if not (rqq is not None):
        raise ValueError("rqq is not None")
    d, w, m = lags
    if rvv.size < m + 30:
        raise ValueError("insufficient history")
    n = rvv.size
    x = np.full((n, 5), np.nan)
    x[:, 0] = 1.0
    for t in range(m, n):
        x[t, 1] = rvv[t - 1]
        x[t, 2] = rvv[t - 1] * np.sqrt(rqq[t - 1])
        x[t, 3] = rvv[max(0, t - w) : t].mean()
        x[t, 4] = rvv[t - m : t].mean()
    valid = np.isfinite(x[:, 1])
    xv, yv = x[valid], rvv[valid]
    coef = np.linalg.lstsq(xv, yv, rcond=None)[0]
    resid = yv - xv @ coef
    se = _nw_se(xv, resid)
    return {
        "coef": coef,  # b0, b_d, b_q, b_w, b_m
        "se": se,
        "fitted": xv @ coef,
        "resid": resid,
        "r2": 1.0 - float(resid @ resid) / float(((yv - yv.mean()) ** 2).sum()),
    }


def har_forecast(
    fit: dict[str, Array | float], rv_history: Array, lags: tuple[int, int, int] = (1, 5, 22)
) -> float:
    """One-step-ahead RV forecast from the most recent rv history."""
    rvv, _ = _check(rv_history)
    d, w, m = lags
    c = np.asarray(fit["coef"], dtype=float)
    if c.size == 4:
        x = np.array([1.0, rvv[-1], rvv[-w:].mean(), rvv[-m:].mean()])
    elif c.size == 5:
        raise ValueError("use harq_forecast for HARQ fits")
    else:
        raise ValueError("unexpected coef length")
    return float(c @ x)


def harq_forecast(
    fit: dict[str, Array | float],
    rv_history: Array,
    rq_history: Array,
    lags: tuple[int, int, int] = (1, 5, 22),
) -> float:
    rvv, rqq = _check(rv_history, rq_history)
    if not (rqq is not None):
        raise ValueError("rqq is not None")
    d, w, m = lags
    c = np.asarray(fit["coef"], dtype=float)
    if c.size != 5:
        raise ValueError("expected a HARQ fit (5 coefs)")
    x = np.array([1.0, rvv[-1], rvv[-1] * np.sqrt(rqq[-1]), rvv[-w:].mean(), rvv[-m:].mean()])
    return float(c @ x)


def bench_har(seed: int = 20261231 + 241) -> dict[str, float]:
    """SYNTHETIC: HAR cascade recovery — β_day/week/month must land
    near the planted (0.3, 0.4, 0.2) and a flat-RV path must give
    near-zero persistence; HARQ must fit the same cascade."""
    rng = np.random.default_rng(seed)
    t, m = 900, 22
    rv = np.full(t, 1.0)
    for i in range(m, t):
        rv[i] += (
            0.3 * rv[i - 1] + 0.4 * float(rv[i - 5 : i].mean()) + 0.2 * float(rv[i - 22 : i].mean())
        )
        rv[i] = max(rv[i] * np.exp(0.25 * rng.normal() - 0.03125), 1e-4)
    fit = har_rv_fit(rv)
    flat = np.exp(rng.normal(0.0, 0.2, 900))
    fit_f = har_rv_fit(flat)
    rq = np.abs(rng.normal(1.0, 0.1, t))
    fitq = harq_fit(rv, rq)
    fc = har_forecast(fit, rv[-80:])
    fcq = harq_forecast(fitq, rv[-80:], rq[-80:])
    coef = np.asarray(fit["coef"])
    checks = [
        abs(float(coef[1]) - 0.3) < 0.2,
        0.05 < float(coef[2]) < 0.75,
        float(fit["r2"]) > 0.3,
        float(fit_f["r2"]) < 0.15,
        np.isfinite(fc) and np.isfinite(fcq) and fc > 0 and fcq > 0,
    ]
    out = {
        "synthetic_har_beta_day": float(coef[1]),
        "synthetic_har_beta_week": float(coef[2]),
        "synthetic_har_r2": float(fit["r2"]),
        "synthetic_har_r2_flat": float(fit_f["r2"]),
        "synthetic_har_forecast": float(fc),
    }
    if not all(checks):
        raise ValueError(f"har off: {out}")
    return out
