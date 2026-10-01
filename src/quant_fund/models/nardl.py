"""Shin-Yu-Greenwood-Nimmo nonlinear ARDL (NARDL) asymmetry model.

References
----------
- Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). "Modelling
  Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear
  ARDL Framework." In *Festschrift in Honor of Peter Schmidt*
  (Sickles & Horrace, eds.), Springer.
- Pesaran, M.H., Shin, Y. & Smith, R.J. (2001). "Bounds Testing
  Approaches to the Analysis of Level Relationships." *Journal of
  Applied Econometrics* 16(3), 289-326.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
The NARDL(p, q) conditional ECM decomposes the regressor's changes
into positive and negative partial sums

    x+_t = x_0 + sum_{i=1..t} max(dx_i, 0),
    x-_t = x_0 + sum_{i=1..t} min(dx_i, 0),

and fits

    dy_t = c + rho y_{t-1} + th+ x+_{t-1} + th- x-_{t-1}
           + sum_j g_j dy_{t-j}
           + sum_j (pi+_j dx+_{t-j} + pi-_j dx-_{t-j}) + e_t,

so the long-run responses are L+ = -th+/rho and L- = -th-/rho.
``asymmetric_multipliers`` propagates a permanent unit step in
each partial-sum component through the fitted ECM recursion
(with dx+ held at 1 for the + channel and analogously for -)
giving the cumulative dynamic multiplier path. ``wald_symmetry``
tests L+ = L- via the delta method on the full OLS covariance,
and ``bounds_f`` reports the PSS-style F-statistic for the joint
restriction rho = th+ = th- = 0 (critical values are user-side;
we report the statistic and the chi-square(3) reference p-value
as an upper bound, which is standard practice as a screening
device). The synth plants asymmetric long-run coefficients and an
I(1) driver; the fit must recover both arms and reject symmetry.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]

# column layout of the design matrix
# [const, y_{t-1}, x+_{t-1}, x-_{t-1}, dy_{t-1..p-1}, dx+_{t..q?}..]
# short-run block: j = 1..p-1 for dy, j = 0..q-1 for dx+/dx-


def partial_sums(dx: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Positive/negative partial sums of a differenced series."""
    dd = np.asarray(dx, dtype=np.float64)
    if dd.ndim != 1 or dd.size < 4 or not np.all(np.isfinite(dd)):
        raise ValueError("bad dx")
    # len-T arrays with a leading 0 so index t is the partial sum
    # through time t (t = 1..T-1); np.diff then gives dx+_t aligned
    # with observation time.
    return (
        np.concatenate([[0.0], np.cumsum(np.maximum(dd, 0.0))]),
        np.concatenate([[0.0], np.cumsum(np.minimum(dd, 0.0))]),
    )


def _design(
    y: FloatArray, xp: FloatArray, xm: FloatArray, p: int, q: int
) -> tuple[FloatArray, FloatArray, int]:
    """NARDL design; returns (dy_rows, X, index_of_pi_blocks)."""
    n_lag = max(p, q) + 1
    dxp = np.diff(xp)
    dxm = np.diff(xm)
    dy_full = np.diff(y)
    n = dy_full.size - n_lag + 1
    if n < 3 * (p + q) + 6:
        raise ValueError("series too short for (p, q)")
    rows = dy_full[n_lag - 1 :]
    cols = [
        np.ones(n),
        y[n_lag - 1 : n_lag - 1 + n],
        xp[n_lag - 1 : n_lag - 1 + n],
        xm[n_lag - 1 : n_lag - 1 + n],
    ]
    for j in range(1, p):
        cols.append(dy_full[n_lag - 1 - j : n_lag - 1 - j + n])
    pi_idx = len(cols)
    for j in range(q):
        cols.append(dxp[n_lag - 1 - j : n_lag - 1 - j + n])
    for j in range(q):
        cols.append(dxm[n_lag - 1 - j : n_lag - 1 - j + n])
    return rows, np.column_stack(cols), pi_idx


def fit_nardl(
    y: FloatArray,
    x: FloatArray,
    p: int = 2,
    q: int = 2,
) -> dict[str, float | FloatArray]:
    """OLS-fit the NARDL(p, q) conditional ECM.

    Raises ValueError on bad shapes, non-finite data, or an
    ill-conditioned design.
    """
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.ndim != 1 or xx.ndim != 1 or yy.size != xx.size:
        raise ValueError("y and x must be equal-length vectors")
    if yy.size < 30 or not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    if p < 1 or q < 1:
        raise ValueError("need p>=1, q>=1")
    dx = np.diff(xx)
    xp, xm = partial_sums(dx)
    dep, des, _pi_idx = _design(yy, xp, xm, p, q)
    beta, *_ = np.linalg.lstsq(des, dep, rcond=None)
    resid = dep - des @ beta
    dof = dep.size - des.shape[1]
    if dof <= 0:
        raise ValueError("insufficient dof")
    s2 = float(resid @ resid / dof)
    xtx_inv = np.linalg.pinv(des.T @ des)
    cov = s2 * xtx_inv
    rho, thp, thm = float(beta[1]), float(beta[2]), float(beta[3])
    if rho >= 0.0:
        raise ValueError("no error correction detected")
    lp, lm = -thp / rho, -thm / rho

    # delta-method SEs for L = -theta/rho: dL = (theta/rho^2, -1/rho)
    def _delta_se(idx_th: int) -> float:
        g = np.zeros(des.shape[1])
        g[1] = beta[idx_th] / (rho * rho)
        g[idx_th] = -1.0 / rho
        v = float(g @ cov @ g)
        return float(np.sqrt(max(v, 0.0)))

    se_lp, se_lm = _delta_se(2), _delta_se(3)
    return {
        "beta": beta,
        "cov": cov,
        "s2": s2,
        "n_obs": float(dep.size),
        "rho": rho,
        "theta_plus": thp,
        "theta_minus": thm,
        "l_plus": lp,
        "l_minus": lm,
        "se_l_plus": se_lp,
        "se_l_minus": se_lm,
        "p": float(p),
        "q": float(q),
    }


def wald_symmetry(fit: dict[str, float | FloatArray]) -> dict[str, float]:
    """Wald test of long-run symmetry L+ = L- (delta method)."""
    rho = float(fit["rho"])
    beta = np.asarray(fit["beta"])
    cov = np.asarray(fit["cov"])
    # R(theta) = -b2/rho + b3/rho = (b3 - b2)/rho; grad wrt (rho, b2, b3)
    g = np.zeros(beta.size)
    g[1] = (beta[2] - beta[3]) / (rho * rho)
    g[2] = -1.0 / rho
    g[3] = 1.0 / rho
    r_val = (beta[3] - beta[2]) / rho
    var = float(g @ cov @ g)
    if var <= 0.0:
        raise ValueError("degenerate covariance")
    w = float(r_val * r_val / var)
    return {
        "wald": w,
        "p_value": float(_stats.chi2.sf(w, df=1)),
        "diff": float(fit["l_plus"] - fit["l_minus"]),
    }


def bounds_f(fit: dict[str, float | FloatArray], y: FloatArray, x: FloatArray) -> dict[str, float]:
    """PSS-style bounds F-statistic for rho = th+ = th- = 0.

    Refits the restricted design (levels terms dropped) on the same
    sample; reports the F-stat and the chi-square(3)-based
    reference p-value (a conservative screening bound, not the
    tabulated PSS critical value).
    """
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    p, q = int(float(fit["p"])), int(float(fit["q"]))
    dx = np.diff(xx)
    xp, xm = partial_sums(dx)
    dep, des, _ = _design(yy, xp, xm, p, q)
    beta_u, *_ = np.linalg.lstsq(des, dep, rcond=None)
    ssr_u = float(np.sum((dep - des @ beta_u) ** 2))
    keep = [0] + list(range(4, des.shape[1]))
    des_r = des[:, keep]
    beta_r, *_ = np.linalg.lstsq(des_r, dep, rcond=None)
    ssr_r = float(np.sum((dep - des_r @ beta_r) ** 2))
    df1, df2 = 3, dep.size - des.shape[1]
    f = (ssr_r - ssr_u) / df1 / (ssr_u / df2)
    f = max(f, 0.0)
    return {
        "f_stat": f,
        "df1": float(df1),
        "df2": float(df2),
        "p_value_ref": float(_stats.chi2.sf(df1 * f, df=df1)),
    }


def asymmetric_multipliers(
    fit: dict[str, float | FloatArray],
    y_last: FloatArray,
    xp_last: float,
    xm_last: float,
    h: int = 20,
) -> dict[str, FloatArray]:
    """Cumulative dynamic multipliers for a permanent unit step.

    ``y_last`` is the trailing window of y of length >= p+1 used to
    seed the recursion; the shock is a permanent +1 step in the
    positive (resp. negative) partial-sum component propagated
    through the fitted ECM with the other channel held at zero.
    """
    if h < 1 or h > 200:
        raise ValueError("bad horizon")
    beta = np.asarray(fit["beta"])
    p, q = int(float(fit["p"])), int(float(fit["q"]))
    yy = np.asarray(y_last, dtype=np.float64)
    if yy.size < p + 1:
        raise ValueError("seed window too short")

    def _path(shock_plus: float, shock_minus: float) -> FloatArray:
        # y_t recursion; x+ path = xp_last + t for + channel.
        y_path = list(yy[-(p + 1) :])
        dy_hist = list(np.diff(yy)[-(p - 1) :]) if p > 1 else []
        m_cum = []
        level0 = y_path[0]
        for t in range(1, h + 1):
            lag_idx = len(y_path) - 1
            ylag = y_path[lag_idx]
            # permanent level step: the partial-sum component jumps
            # by `shock` once (at t=1) and stays elevated.
            xpl = xp_last + shock_plus
            xml = xm_last + shock_minus
            dy = beta[0] + beta[1] * ylag + beta[2] * xpl + beta[3] * xml
            for j in range(1, p):
                if len(dy_hist) - j >= 0:
                    dy += beta[3 + j] * dy_hist[-j]
            # short-run dx terms fire only at the shock date (t=1)
            base = 3 + (p - 1)
            for j in range(q):
                dxp_lag = shock_plus if (t - 1 - j) == 0 else 0.0
                dy += beta[base + j] * dxp_lag
            base2 = base + q
            for j in range(q):
                dxm_lag = shock_minus if (t - 1 - j) == 0 else 0.0
                dy += beta[base2 + j] * dxm_lag
            dy_hist.append(dy)
            y_path.append(y_path[-1] + dy)
            m_cum.append(y_path[-1] - (level0 + xp_last * 0.0))
        # subtract the no-shock baseline path for clean multipliers
        return np.asarray(m_cum)

    def _baseline() -> FloatArray:
        return _path(0.0, 0.0)

    base = _baseline()
    m_plus = _path(1.0, 0.0) - base
    m_minus = _path(0.0, 1.0) - base
    return {"m_plus": m_plus, "m_minus": m_minus}


def synth_nardl(
    seed: int = 20261231 + 299,
    t: int = 300,
    l_plus: float = 0.9,
    l_minus: float = -0.4,
    rho: float = -0.15,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC NARDL-consistent DGP with I(1) driver."""
    rng = np.random.default_rng(seed)
    dx = rng.standard_normal(t) * 0.5
    x = np.cumsum(dx)
    xp, xm = partial_sums(dx[1:])
    y = np.empty(t)
    y[0] = l_plus * xp[0] + l_minus * xm[0]
    thp, thm = -l_plus * rho, -l_minus * rho
    for i in range(1, t):
        ecm = rho * y[i - 1] + thp * xp[i - 1] + thm * xm[i - 1]
        dxp = xp[i] - xp[i - 1]
        dxm = xm[i] - xm[i - 1]
        dy = ecm + 0.35 * dxp - 0.2 * dxm + rng.standard_normal() * 0.2
        y[i] = y[i - 1] + dy
    return {
        "y": y,
        "x": x,
        "l_plus": l_plus,
        "l_minus": l_minus,
        "rho": rho,
    }


def bench_nardl(seed: int = 20261231 + 299) -> dict[str, float]:
    """Wave-52 self-check: recovers asymmetric long-run arms."""
    d = synth_nardl(seed=seed)
    fit = fit_nardl(np.asarray(d["y"]), np.asarray(d["x"]), p=2, q=2)
    w = wald_symmetry(fit)
    bf = bounds_f(fit, np.asarray(d["y"]), np.asarray(d["x"]))
    lp_err = abs(float(fit["l_plus"]) - float(d["l_plus"]))
    lm_err = abs(float(fit["l_minus"]) - float(d["l_minus"]))
    ok = lp_err < 0.3 and lm_err < 0.3 and w["p_value"] < 0.05 and bf["f_stat"] > 3.0
    return {
        "l_plus": float(fit["l_plus"]),
        "l_minus": float(fit["l_minus"]),
        "lp_err": lp_err,
        "lm_err": lm_err,
        "wald_sym": w["wald"],
        "pval_sym": w["p_value"],
        "f_bounds": bf["f_stat"],
        "score": float(ok),
    }
