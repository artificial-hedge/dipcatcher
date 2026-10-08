"""Engle-Ng (1993) sign/size-bias news-impact asymmetry tests.

References
----------
- Engle, R.F. & Ng, V.K. (1993). "Measuring and Testing the
  Impact of News on Volatility." *Journal of Finance* 48(5),
  1749-1778.
- Engle, R.F. (1982). "Autoregressive Conditional
  Heteroscedasticity with Estimates of the Variance of United
  Kingdom Inflation." *Econometrica* 50(4), 987-1007.
- Ng, V.K. & Pirone, J. (1996). "A Study of the Power of the
  Sign and Size Bias Tests." UCSD WP.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Engle-Ng diagnostic for whether a volatility model captures
the leverage/asymmetry of the news impact curve. Given
standardized residuals ``z_t = r_t / sigma_t`` (sigma from any
fitted vol model — the test diagnoses whether it *misses*
asymmetry), three moment conditions should hold under
correct specification:

    sign bias:           E[I(z<0) z^2] vs a + b I(z_{t-1}<0)  — b = 0
    negative size bias:  z^2 = a + c I(z_{t-1}<0) |z_{t-1}|  — c = 0
    positive size bias:  z^2 = a + d I(z_{t-1}>0) |z_{t-1}|  — d = 0
    joint LM             ~ chi2(3) over the four regressors.

We implement the joint regression
``z^2 = a + b S^- + c S^- |z| + d S^+ |z| + e`` and the LM
statistic ``T*R^2 ~ chi2(3)`` plus each coefficient's t-test.
``synth_engle_ng`` draws an EGARCH-style leverage DGP
(negative returns raise vol) for the alternative and a plain
GARCH(1,1) (symmetric) for the null; the test is run on raw
returns — finding asymmetry where planted and none where
absent.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def engle_ng_test(z: FloatArray) -> dict[str, float]:
    """Joint Engle-Ng sign/size-bias LM test on residuals z."""
    zz = np.asarray(z, dtype=np.float64)
    if zz.ndim != 1 or zz.size < 200 or not np.all(np.isfinite(zz)):
        raise ValueError("bad residuals")
    if np.std(zz) < 1e-12:
        raise ValueError("degenerate")
    z_lag = zz[:-1]
    y = zz[1:] ** 2
    s_neg = (z_lag < 0).astype(np.float64)
    s_pos = 1.0 - s_neg
    x = np.column_stack([np.ones(y.size), s_neg, s_neg * np.abs(z_lag), s_pos * np.abs(z_lag)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    t = y.size
    k = 3
    s2 = float(resid @ resid / (t - k - 1))
    cov = s2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    tstat = coef / se
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
    lm = t * r2
    return {
        "b_sign": float(coef[1]),
        "t_sign": float(tstat[1]),
        "t_neg_size": float(tstat[2]),
        "t_pos_size": float(tstat[3]),
        "r2": r2,
        "lm": float(lm),
        "p_joint": float(_stats.chi2.sf(lm, k)),
        "p_sign": float(2.0 * _stats.t.sf(abs(tstat[1]), t - k - 1)),
    }


def _garch11_filter(r: FloatArray) -> FloatArray:
    """Symmetric GARCH(1,1) QMLE -> sigma path (misspec on leverage)."""
    v0 = float(np.var(r))

    def nll(th: FloatArray) -> float:
        w, a, b = float(th[0]), float(th[1]), float(th[2])
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        h = np.empty(r.size)
        h[0] = v0
        for t in range(1, r.size):
            h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        if not np.all(np.isfinite(h)) or np.any(h <= 0):
            return 1e12
        return float(0.5 * np.sum(np.log(h) + r * r / h))

    res = _opt.minimize(
        nll,
        np.array([0.05 * v0, 0.1, 0.8]),
        method="Nelder-Mead",
        options={"maxiter": 400},
    )
    w, a, b = float(res.x[0]), float(res.x[1]), float(res.x[2])
    h = np.empty(r.size)
    h[0] = v0
    for t in range(1, r.size):
        h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
    return np.asarray(np.sqrt(np.maximum(h, 1e-16)))


def synth_engle_ng(
    seed: int = 20261231 + 323,
    n: int = 2000,
    gamma: float = -0.6,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC: EGARCH-leverage alt vs symmetric GARCH null."""
    rng = np.random.default_rng(seed)
    # leverage: log-var loads on -z_{t-1} asymmetrically
    r = np.zeros(n)
    logv = np.zeros(n)
    for t in range(1, n):
        z_prev = r[t - 1] / np.sqrt(np.exp(logv[t - 1]) + 1e-12)
        logv[t] = 0.9 * logv[t - 1] + 0.1 * (abs(z_prev) - 0.8) + gamma * z_prev * 0.5
        r[t] = np.exp(0.5 * logv[t]) * rng.normal()
    # symmetric GARCH(1,1)
    r2 = np.zeros(n)
    h = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.1 * r2[t - 1] ** 2 + 0.85 * h[t - 1]
        r2[t] = np.sqrt(h[t]) * rng.normal()
    return np.asarray(r[200:]), np.asarray(r2[200:])


def bench_engle_ng(
    seed: int = 20261231 + 323,
) -> dict[str, float]:
    """Wave-55 self-check: leverage detected under symmetric fit."""
    lev, sym = synth_engle_ng(seed=seed)
    z_lev = lev / _garch11_filter(lev)
    z_sym = sym / _garch11_filter(sym)
    r_lev = engle_ng_test(z_lev)
    r_sym = engle_ng_test(z_sym)
    ok = r_lev["p_joint"] < 0.01 and r_sym["p_joint"] > 0.01
    return {
        "synthetic_lm_lev": r_lev["lm"],
        "synthetic_p_lev": r_lev["p_joint"],
        "synthetic_p_sym": r_sym["p_joint"],
        "synthetic_b_sign": r_lev["b_sign"],
        "synthetic_r2_lev": r_lev["r2"],
        "synthetic_score": float(ok),
    }
