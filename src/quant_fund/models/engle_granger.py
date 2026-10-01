"""Engle-Granger (1987) + Phillips-Ouliaris (1990) cointegration.

References
----------
- Engle, R.F. & Granger, C.W.J. (1987). "Co-integration and
  Error Correction: Representation, Estimation, and Testing."
  *Econometrica* 55(2), 251-276.
- Phillips, P.C.B. & Ouliaris, S. (1990). "Asymptotic Properties
  of Residual Based Tests for Cointegration." *Econometrica*
  58(1), 165-193.
- MacKinnon, J.G. (1991/2010). "Critical Values for
  Cointegration Tests." Queen's Economics Dept WP 1227.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The two-step procedure: (1) regress ``y`` on ``x`` (levels,
constant included) and keep the residual ``u``; (2) test
``u`` for a unit root — cointegration iff residuals are
stationary. Step 2 uses the Engle-Granger/Phillips-Ouliaris
tau statistic ``t = rho_tilde / se`` on the AR(1) residual
regression *without* intercept, compared against the
MacKinnon cointegration-response-surface critical values for
``n=2`` variables (``c(beta) = phi_inf + phi_1 T^{-1} +
phi_2 T^{-2}`` with ``(-3.900, -10.534, -30.03)`` at 5% for
the no-trend n=2 case). Because residuals are estimated, the
ordinary ADF critical values are invalid — the MacKinnon
surface is what makes this Engle-Granger rather than a raw
ADF. ``ecm_fit`` then fits the error-correction model
``dy = alpha * u_{t-1} + phi * dx + e`` and reports the
adjustment speed ``alpha``, which must be negative and
significant under cointegration. The synth cointegrates a
common random walk plus stationary noise against two
independent walks as the null.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _ols(y: FloatArray, x: FloatArray) -> tuple[float, float, FloatArray]:
    """y = a + b x + u; returns (a, b, u)."""
    X = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), np.asarray(y - X @ coef)


def eg_tau(y: FloatArray, x: FloatArray) -> dict[str, float]:
    """Engle-Granger/Phillips-Ouliaris residual tau statistic."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.shape != xx.shape or yy.ndim != 1 or yy.size < 50:
        raise ValueError("bad series")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("non-finite")
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        raise ValueError("degenerate series")
    _, beta, u = _ols(yy, xx)
    # AR(1) on residuals, no intercept: du = (rho-1) u_{t-1} + e
    du = np.diff(u)
    lag = u[:-1]
    sxx = float(lag @ lag)
    if sxx <= 1e-20:
        raise ValueError("degenerate residuals")
    rho_m1 = float(lag @ du / sxx)
    resid = du - rho_m1 * lag
    se = float(np.sqrt(np.sum(resid**2) / (du.size - 1) / sxx))
    tau = rho_m1 / se
    # MacKinnon response surface, no-trend n=2, 5% critical value
    t = yy.size
    crit = -3.3377 - 5.967 / t - 8.98 / (t * t)
    # (Engle-Yoo/MacKinnon 1991 table: -3.3377 -5.967 -8.98 for n=2)
    return {
        "tau": float(tau),
        "beta": float(beta),
        "crit5": float(crit),
        "resid_var": float(np.var(u)),
    }


def ecm_fit(y: FloatArray, x: FloatArray) -> dict[str, float]:
    """Error-correction adjustment speed ``alpha`` on lagged resid."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.shape != xx.shape or yy.ndim != 1 or yy.size < 50:
        raise ValueError("bad series")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("non-finite")
    _, _, u = _ols(yy, xx)
    dy = np.diff(yy)
    dx = np.diff(xx)
    X = np.column_stack([u[:-1], dx])
    coef, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ coef
    dof = dy.size - 2
    if dof < 2:
        raise ValueError("too short")
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_a = float(np.sqrt(cov[0, 0]))
    alpha = float(coef[0])
    return {
        "alpha": alpha,
        "alpha_t": alpha / se_a,
        "alpha_p": float(2.0 * _stats.t.sf(abs(alpha / se_a), dof)),
    }


def synth_eg(
    seed: int = 20261231 + 318,
    n: int = 500,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC cointegrated pair vs independent null pair."""
    rng = np.random.default_rng(seed)
    w = np.cumsum(rng.normal(0.0, 1.0, n))
    y1 = w + rng.normal(0.0, 0.3, n)
    x1 = 0.8 * w + rng.normal(0.0, 0.3, n)
    y0 = np.cumsum(rng.normal(0.0, 1.0, n))
    x0 = np.cumsum(rng.normal(0.0, 1.0, n))
    return np.asarray(y1), np.asarray(x1), np.asarray(y0), np.asarray(x0)


def bench_engle_granger(
    seed: int = 20261231 + 318,
) -> dict[str, float]:
    """Wave-55 self-check: cointegrated pair rejected, null passed."""
    y1, x1, y0, x0 = synth_eg(seed=seed)
    r_ci = eg_tau(y1, x1)
    r_nc = eg_tau(y0, x0)
    ecm = ecm_fit(y1, x1)
    ok = (
        r_ci["tau"] < r_ci["crit5"]
        and r_nc["tau"] > r_nc["crit5"]
        and ecm["alpha"] < 0
        and ecm["alpha_p"] < 0.05
    )
    return {
        "tau_ci": r_ci["tau"],
        "crit5": r_ci["crit5"],
        "tau_nc": r_nc["tau"],
        "alpha": ecm["alpha"],
        "alpha_p": ecm["alpha_p"],
        "score": float(ok),
    }
