"""Predictive regression with Stambaugh bias and Bonferroni bounds (SYNTHETIC).

y_t = a + b*x_{t-1} + u_t with persistent, endogenous regressor x
(x_t = rho*x_{t-1} + v_t, corr(u,v) < 0). The OLS slope is biased upward
by Stambaugh (1999): E[b_hat - b] ~ -(1 + 3*rho)/T * phi where
phi = cov(u, v)/var(v) is the AR bias of rho-hat.

- ``stambaugh_correct``: bias-corrected slope using the fitted AR(1) on x.
- ``bonferroni_test``: Campbell-Yogo style Bonferroni bound on the slope
  CI — the union bound over the CI of rho.
- ``long_horizon_predict``: overlapping h-step-ahead regression with
  Newey-West t-stat on the slope.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _check(x: Array, y: Array) -> tuple[Array, Array]:
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    if xx.size != yy.size or xx.size < 20:
        raise ValueError("x and y must be equal-length with n >= 20")
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("non-finite input")
    if xx.std() == 0:
        raise ValueError("degenerate regressor")
    return xx, yy


def _ar1(x: Array) -> tuple[float, Array]:
    xc = x - x.mean()
    denom = float(xc[:-1] @ xc[:-1])
    if denom == 0:
        raise ValueError("degenerate AR input")
    rho = float(xc[1:] @ xc[:-1] / denom)
    return rho, xc[1:] - rho * xc[:-1]


def _ols_xy(x: Array, y: Array) -> tuple[float, float, Array]:
    xd = np.column_stack([np.ones(x.size), x])
    beta = np.linalg.lstsq(xd, y, rcond=None)[0]
    resid = y - xd @ beta
    return float(beta[0]), float(beta[1]), resid


def stambaugh_correct(x: Array, y: Array) -> dict[str, float]:
    """Stambaugh-corrected slope.

    Reports the raw slope, rho, the implied phi, and the corrected slope
    b_bc = b_hat + (1 + 3*rho)/T * phi.
    """
    xx, yy = _check(x, y)
    a, b, u = _ols_xy(xx[:-1], yy[1:])
    rho, v = _ar1(xx)
    if v.size < 4:
        raise ValueError("AR residual too short")
    phi = float(np.cov(u, v)[0, 1] / np.var(v))
    bias = (1.0 + 3.0 * rho) / xx.size * phi
    return {
        "b_ols": b,
        "rho": rho,
        "phi": phi,
        "b_bc": b + bias,
        "bias": bias,
        "intercept": a,
    }


def bonferroni_test(x: Array, y: Array, alpha: float = 0.05) -> dict[str, float | bool]:
    """Campbell-Yogo Bonferroni bound for the slope CI.

    Build the (1-alpha/2) CI for rho from the AR(1) estimate, then re-fit
    y on the rho-implied innovation of x for each bound and return the
    widest slope interval.
    """
    xx, yy = _check(x, y)
    rho, _ = _ar1(xx)
    n = xx.size
    se_rho = np.sqrt((1 - rho**2) / n)
    z = stats.norm.ppf(1 - alpha / 4.0)  # Bonferroni split
    lo = float(rho - z * se_rho)
    hi = float(min(rho + z * se_rho, 0.9999))
    intervals = []
    for r in (lo, rho, hi):
        x_innov = xx[1:] - r * xx[:-1]
        _, b, u = _ols_xy(x_innov, yy[1:])
        se_b = float(np.sqrt(np.sum(u * u) / max(u.size - 2, 1) / np.sum(x_innov**2)))
        intervals.append((b - z * se_b, b + z * se_b))
    lo_b = min(iv[0] for iv in intervals)
    hi_b = max(iv[1] for iv in intervals)
    return {
        "ci_lo": lo_b,
        "ci_hi": hi_b,
        "rho": rho,
        "rho_ci": (lo, hi),
        "significant": bool(lo_b > 0 or hi_b < 0),
        "alpha": alpha,
    }


def long_horizon_predict(
    x: Array,
    y: Array,
    h: int = 4,
    nw_lags: int | None = None,
) -> dict[str, float]:
    """y_{t+h} aggregated over h periods on x_t, with Newey-West t on slope."""
    xx, yy = _check(x, y)
    n = xx.size
    if h < 1 or h >= n - 4:
        raise ValueError("h must satisfy 1 <= h < n - 4")
    if nw_lags is None:
        nw_lags = h
    if nw_lags < 0 or nw_lags >= n - h:
        raise ValueError("nw_lags out of range")
    yh = np.array([yy[i + 1 : i + 1 + h].sum() for i in range(n - h)])
    xt = xx[: n - h]
    a, b, u = _ols_xy(xt, yh)
    # Newey-West meat: sum_l w_l * sum_t e_t e_{t-l} x_t x_{t-l}
    ex = u * xt
    meat = float(ex @ ex)
    for lag in range(1, nw_lags + 1):
        w = 1.0 - lag / (nw_lags + 1.0)
        meat += 2.0 * w * float(ex[lag:] @ ex[:-lag])
    xxd = float(xt @ xt)
    se_b = np.sqrt(meat / (xxd * xxd)) if xxd > 0 else np.nan
    t_stat = b / se_b if se_b > 0 else np.nan
    return {
        "b": b,
        "se_b": float(se_b),
        "t": float(t_stat),
        "p_two_sided": float(2 * stats.norm.sf(abs(t_stat)))
        if np.isfinite(t_stat)
        else float("nan"),
        "intercept": a,
        "h": float(h),
        "n_used": float(xt.size),
    }


def bench_predictive_regression(seed: int = 20261231 + 370) -> dict[str, float]:
    """Persistent-regressor oracle: on an AR(1) predictor + correlated-
    innovation DGP, the Stambaugh-corrected slope must sit closer to the
    true β than the raw OLS slope; a flat DGP is not 'significant'."""
    rng = np.random.default_rng(seed)
    n, rho, beta = 400, 0.97, 0.15
    # x_t = rho x_{t-1} + v_t; y_{t+1} = beta x_t + u, corr(u, v) < 0
    v = rng.standard_normal(n)
    u = -0.7 * v + np.sqrt(1 - 0.7**2) * rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + v[t]
    y = np.zeros(n)
    y[1:] = beta * x[:-1] + u[1:]
    st = stambaugh_correct(x, y)
    bf = bonferroni_test(x, y)
    lh = long_horizon_predict(x, y, h=4)
    # null: independent returns → no significance
    y0 = rng.standard_normal(n)
    bf0 = bonferroni_test(x, y0)
    # Stambaugh reduces *expected* bias: mean |b_bc - β| over sims beats OLS
    errs_ols = []
    errs_bc = []
    for s in range(12):
        r2 = np.random.default_rng(seed + 1 + s)
        vv = r2.standard_normal(n)
        uu = -0.7 * vv + np.sqrt(1 - 0.7**2) * r2.standard_normal(n)
        xx = np.zeros(n)
        for t in range(1, n):
            xx[t] = rho * xx[t - 1] + vv[t]
        yy = np.zeros(n)
        yy[1:] = beta * xx[:-1] + uu[1:]
        s2 = stambaugh_correct(xx, yy)
        errs_ols.append(abs(float(s2["b_ols"]) - beta))
        errs_bc.append(abs(float(s2["b_bc"]) - beta))
    checks = [
        float(np.mean(errs_bc)) < float(np.mean(errs_ols)),
        float(st["bias"]) != 0.0,
        bool(bf["significant"]),
        not bool(bf0["significant"]),
        float(lh["p_two_sided"]) < 0.05,
    ]
    if not all(checks):
        raise ValueError("predictive-regression oracle checks failed")
    return {
        "synthetic_pr_ols_err": float(np.mean(errs_ols)),
        "synthetic_pr_bc_err": float(np.mean(errs_bc)),
        "synthetic_pr_bias": float(st["bias"]),
        "synthetic_pr_lh_p": float(lh["p_two_sided"]),
        "synthetic_pr_detects": float(bool(bf["significant"]) and not bool(bf0["significant"])),
        "synthetic_pr_score": float(sum(checks) / len(checks)),
    }
