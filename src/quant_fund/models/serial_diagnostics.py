"""Serial-correlation diagnostics for regression
residuals and raw series.

Canonical references:

- Durbin & Watson (1950/51) 'Testing for serial
  correlation in least squares regression' Biometrika
  — DW = sum(de^2)/sum(e^2), ~2 under independence.
  Exact bounds-dependent: we report the statistic and
  an approximate normal p via the Durbin-Watson
  mean/variance under independence.
- Durbin (1970) 'Testing for serial correlation in
  least-squares regression when some of the regressors
  are lagged dependent variables' Econometrica 38 —
  Durbin's h = rho_hat sqrt(n/(1-n Var(b_lag))).
- Breusch (1978) 'Testing for autocorrelation in
  dynamic linear models' AER 10 and Godfrey (1978)
  Econometrica 46 — LM = n R^2 of e on [x, e_{-1..-p}]
  ~ chi2_p.
- Ljung & Box (1978) 'On a measure of lack of fit in
  time series models' Biometrika 65 —
  Q = n(n+2) sum r_k^2/(n-k) ~ chi2_m.

`bench_serial`: AR(0.7) residuals detected by all;
white noise accepted ~95%.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _check(e: FloatArray) -> FloatArray:
    ea = np.asarray(e, dtype=np.float64).ravel()
    if ea.size < 20 or not np.isfinite(ea).all():
        raise ValueError("bad residual series")
    return ea


def durbin_watson(e: FloatArray) -> dict[str, float]:
    """DW statistic and approximate p (vs positive AR(1))."""
    ea = _check(e)
    de = np.diff(ea)
    dw = float((de**2).sum() / (ea**2).sum())
    # Approximation: DW ~ 2(1-r1); use r1 z-scored by
    # Bartlett se for the directionless two-sided p.
    r1 = float((ea[:-1] * ea[1:]).sum() / (ea**2).sum())
    z = r1 * np.sqrt(ea.size)
    return {
        "stat": dw,
        "r1": r1,
        "pvalue": float(2 * min(stats.norm.sf(z), stats.norm.cdf(z))),
    }


def durbin_h(e: FloatArray, var_b_lag: float) -> dict[str, float]:
    """Durbin's h for models with a lagged dependent
    regressor. Falls back to the m-test (sqrt(n) r1)
    when 1 - n Var(b_lag) <= 0."""
    ea = _check(e)
    n = ea.size
    r1 = float((ea[:-1] * ea[1:]).sum() / (ea**2).sum())
    denom = 1.0 - n * var_b_lag
    if denom > 0:
        h = r1 * np.sqrt(n / denom)
    else:
        h = r1 * np.sqrt(n)
    return {
        "stat": float(h),
        "pvalue": float(2 * stats.norm.sf(abs(h))),
        "used_m": bool(denom <= 0),
    }


def breusch_godfrey(e: FloatArray, x: FloatArray, lags: int = 2) -> dict[str, float]:
    """BG LM: regress e on x and lagged residuals."""
    ea = _check(e)
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    n = ea.size - lags
    cols = [np.ones(n)]
    for j in range(xa.shape[1]):
        cols.append(xa[lags:, j])
    for k in range(1, lags + 1):
        cols.append(ea[lags - k : n + lags - k])
    z = np.column_stack(cols)
    y = ea[lags:]
    b = np.linalg.lstsq(z, y, rcond=None)[0]
    ss = float(((y - z @ b) ** 2).sum())
    s0 = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss / s0 if s0 > 0 else 0.0
    lm = n * r2
    return {
        "stat": float(lm),
        "pvalue": float(stats.chi2.sf(lm, lags)),
        "df": float(lags),
    }


def ljung_box(e: FloatArray, m: int = 10, model_df: int = 0) -> dict[str, float]:
    """Ljung-Box Q on the first m residual autocorrelations."""
    ea = _check(e)
    n = ea.size
    ec = ea - ea.mean()
    s2 = float((ec**2).sum())
    r = np.array([(ec[k:] * ec[: n - k]).sum() / s2 for k in range(1, m + 1)])
    q = n * (n + 2) * float((r**2 / (n - np.arange(1, m + 1))).sum())
    df = m - model_df
    return {
        "stat": float(q),
        "pvalue": float(stats.chi2.sf(q, max(df, 1))),
        "df": float(max(df, 1)),
    }


def bench_serial(seed: int = 527) -> dict[str, float]:
    """SYNTHETIC: y = a+bx + e, e = 0.7 e_{-1} + u AR(1)
    residual detected by DW, BG, LB; iid control."""
    rng = np.random.default_rng(seed)
    R = 60
    rej_null = {"dw": 0, "bg": 0, "lb": 0}
    rej_alt = {"dw": 0, "bg": 0, "lb": 0}
    for _ in range(R):
        x = rng.normal(0, 1, 150)
        e_iid = rng.normal(0, 1, 150)
        e_ar = np.zeros(150)
        for t in range(1, 150):
            e_ar[t] = 0.7 * e_ar[t - 1] + rng.normal(0, 1)
        xd = np.column_stack([np.ones(150), x])
        for tag, ee in (("null", e_iid), ("alt", e_ar)):
            y = 1 + 0.5 * x + ee
            b = np.linalg.lstsq(xd, y, rcond=None)[0]
            res = y - xd @ b
            pv_dw = durbin_watson(res)["pvalue"]
            pv_bg = breusch_godfrey(res, x[:, None], lags=2)["pvalue"]
            pv_lb = ljung_box(res, m=10, model_df=1)["pvalue"]
            if tag == "null":
                rej_null["dw"] += int(pv_dw < 0.05)
                rej_null["bg"] += int(pv_bg < 0.05)
                rej_null["lb"] += int(pv_lb < 0.05)
            else:
                rej_alt["dw"] += int(pv_dw < 0.05)
                rej_alt["bg"] += int(pv_bg < 0.05)
                rej_alt["lb"] += int(pv_lb < 0.05)
    out: dict[str, float] = {}
    for k in rej_null:
        out[f"synthetic_type1_{k}"] = rej_null[k] / R
        out[f"synthetic_power_{k}"] = rej_alt[k] / R
        if rej_alt[k] < R * 0.8:
            raise ValueError(f"{k} misses AR(1)")
        if rej_null[k] > R * 0.18:
            raise ValueError(f"{k} over-rejects")
    return out
