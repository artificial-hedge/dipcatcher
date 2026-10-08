"""Pooled mean group (PMG) + mean group (MG) panel ARDL estimation.

References
----------
- Pesaran, M.H., Shin, Y. & Smith, R.P. (1999). "Pooled Mean Group
  Estimation of Dynamic Heterogeneous Panels." *Journal of the
  American Statistical Association* 94(446), 621-634.
- Pesaran, M.H. & Smith, R.P. (1995). "Estimating Long-Run
  Relationships from Dynamic Heterogeneous Panels." *Journal of
  Econometrics* 68(1), 79-113.
- Blackburne, E.F. & Frank, M.W. (2007). "Estimation of Nonstationary
  Heterogeneous Panels." *Stata Journal* 7(2), 197-208.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
ARDL(p, q) in error-correction form:

    d y_it = phi_i (y_{i,t-1} - theta' x_{i,t-1})
             + sum_j a_ij d y_{i,t-j} + sum_j b_ij d x_{i,t-j}
             + mu_i + e_it,

with long-run coefficients ``theta`` common across groups and
adjustment speeds ``phi_i`` / intercepts ``mu_i`` group-specific.
The PMG estimate minimizes the pooled SSR over ``theta``: given
theta, each group's design is linear in (phi_i, a_ij, b_ij, mu_i),
so the concentrated objective is separable per group and solved by
``scipy.optimize.least_squares``. The MG comparator averages
per-group ARDL coefficient vectors (Pesaran-Smith 1995). The synth
plants one shared long-run theta with heterogeneous adjustment
speeds — PMG must recover theta tighter than MG's raw average.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt

FloatArray = NDArray[np.float64]


def _ecm_design(
    y: FloatArray,
    x: FloatArray,
    theta: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Per-group ECM regression for one panel unit.

    ``y`` (T,) and ``x`` (T, k); returns dep ``dy_t`` and design
    ``[y_{t-1} - theta'x_{t-1}, dy_{t-1}, dx_{t-1}, 1]`` over rows
    ``t = 2..T-1`` — one lag of each difference, matching the
    ARDL(1,1) error-correction representation.
    """
    t_ = y.size
    dy = np.diff(y)
    dx = np.diff(x, axis=0)
    ec = y[1 : t_ - 1] - x[1 : t_ - 1, :] @ theta
    cols = [ec[:, None], dy[: t_ - 2, None], dx[: t_ - 2, :], np.ones((t_ - 2, 1))]
    dep = dy[1:]
    return dep, np.hstack(cols)


def _group_ssr(
    y: FloatArray,
    x: FloatArray,
    theta: FloatArray,
) -> tuple[float, FloatArray, float]:
    dep, xx = _ecm_design(y, x, theta)
    beta, *_ = np.linalg.lstsq(xx, dep, rcond=None)
    resid = dep - xx @ beta
    dof = max(dep.size - xx.shape[1], 1)
    return float(resid @ resid), beta, float(dof)


def pmg_ardl(
    y_groups: FloatArray,
    x_groups: FloatArray,
    theta_init: FloatArray | None = None,
    phi_true_prior: float | None = None,
) -> dict[str, float | FloatArray]:
    """Pooled-mean-group long-run estimator.

    ``y_groups`` (N, T) and ``x_groups`` (N, T, k) hold N panel
    units. Returns the common long-run vector ``theta``, the
    mean/median adjustment speed across groups, the pooled SSR, and
    per-group phi.
    """
    yy = np.asarray(y_groups, dtype=np.float64)
    xx = np.asarray(x_groups, dtype=np.float64)
    if yy.ndim != 2 or xx.ndim != 3 or xx.shape[:2] != yy.shape:
        raise ValueError("y (N,T) and x (N,T,k) shapes required")
    n, t, k = yy.shape[0], yy.shape[1], xx.shape[2]
    if n < 2 or t < 12 or k < 1:
        raise ValueError("panel too small")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("non-finite panel")
    if theta_init is None:
        theta_init = np.zeros(k)
    th0 = np.asarray(theta_init, dtype=np.float64)
    if th0.shape != (k,) or not np.all(np.isfinite(th0)):
        raise ValueError("theta_init shape mismatch")

    def pooled_ssr(th: FloatArray) -> float:
        total = 0.0
        for i in range(n):
            s, _, _ = _group_ssr(yy[i], xx[i], th)
            total += s
        return total

    res = _opt.least_squares(
        lambda th: np.asarray([pooled_ssr(th)]),
        th0,
        method="lm",
    )
    theta = np.asarray(res.x, dtype=np.float64)
    phis = np.empty(n)
    ssrs = np.empty(n)
    for i in range(n):
        s, beta, dof = _group_ssr(yy[i], xx[i], theta)
        ssrs[i] = s / dof
        phis[i] = float(beta[0])
    return {
        "theta": theta,
        "phi_mean": float(np.mean(phis)),
        "phi_median": float(np.median(phis)),
        "phi_sd": float(np.std(phis)),
        "ssr_pooled": float(np.sum(ssrs)),
        "cost": float(res.cost),
        "optimality": float(res.optimality),
        "n_groups": float(n),
        "k": float(k),
        "phis": phis,
    }


def mean_group(
    y_groups: FloatArray,
    x_groups: FloatArray,
) -> dict[str, float | FloatArray]:
    """Pesaran-Smith mean-group estimator: average per-group ARDL.

    Fits each unit's ARDL(1,1) ECM separately, converts to the
    implied long-run ``theta_i = -b_ec / phi_i`` per unit, then
    averages with cross-group standard errors.
    """
    yy = np.asarray(y_groups, dtype=np.float64)
    xx = np.asarray(x_groups, dtype=np.float64)
    if yy.ndim != 2 or xx.ndim != 3 or xx.shape[:2] != yy.shape:
        raise ValueError("y (N,T) and x (N,T,k) shapes required")
    n, t, k = yy.shape[0], yy.shape[1], xx.shape[2]
    if n < 2 or t < 12:
        raise ValueError("panel too small")
    thetas = np.empty((n, k))
    phis = np.empty(n)
    for i in range(n):
        # unrestricted per-group ECM: dy on [y_{t-1}, x_{t-1}, dy lag, dx lag, 1]
        dy = np.diff(yy[i])
        dx = np.diff(xx[i], axis=0)
        cols = [
            yy[i][1 : t - 1, None],
            xx[i][1 : t - 1, :],
            dy[: t - 2, None],
            dx[: t - 2, :],
            np.ones((t - 2, 1)),
        ]
        dep = dy[1:]
        xd = np.hstack(cols)
        beta, *_ = np.linalg.lstsq(xd, dep, rcond=None)
        phi_i = float(beta[0])
        if abs(phi_i) < 1e-8:
            phi_i = -1e-8
        phis[i] = phi_i
        thetas[i] = -beta[1 : 1 + k] / phi_i
    theta_mg = np.mean(thetas, axis=0)
    theta_se = np.std(thetas, axis=0, ddof=1) / np.sqrt(n)
    return {
        "theta_mg": theta_mg,
        "theta_se": theta_se,
        "theta_i": thetas,
        "phi_mean": float(np.mean(phis)),
        "n_groups": float(n),
    }


def synth_pmg(
    seed: int = 20261231 + 295,
    n: int = 8,
    t: int = 160,
    k: int = 1,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC cointegrated panel: common theta, heterogeneous phi.

    Each unit follows the ECM ``dy = phi_i (y - theta'x) + mu_i + e``
    with x a unit-specific random walk — a genuine long-run
    relationship under heterogeneous short-run dynamics.
    """
    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(0.4, 0.9, k)
    phi_true = -rng.uniform(0.1, 0.5, n)
    mu_true = rng.normal(0.0, 0.05, n)
    yy = np.empty((n, t))
    xx = np.empty((n, t, k))
    for i in range(n):
        xi = np.cumsum(rng.normal(0.0, 1.0, (t, k)), axis=0) + rng.normal(0, 1, k)
        xx[i] = xi
        yi = np.empty(t)
        yi[0] = xi[0] @ theta_true
        for s in range(1, t):
            ec = yi[s - 1] - xi[s - 1] @ theta_true
            yi[s] = (
                yi[s - 1]
                + phi_true[i] * ec
                + mu_true[i]
                + rng.normal(0.0, 0.5)
                + (xi[s] - xi[s - 1]) @ (0.3 * theta_true)
            )
        yy[i] = yi
    return {
        "y_groups": yy,
        "x_groups": xx,
        "theta_true": theta_true,
        "phi_true": phi_true,
    }


def bench_pmg_ardl(seed: int = 20261231 + 295) -> dict[str, float]:
    """Wave-51 self-check: PMG recovers the planted long-run theta."""
    d = synth_pmg(seed=seed)
    yy = np.asarray(d["y_groups"])
    xx = np.asarray(d["x_groups"])
    th_t = np.asarray(d["theta_true"])
    pmg = pmg_ardl(yy, xx)
    mg = mean_group(yy, xx)
    th_p = np.asarray(pmg["theta"])
    th_m = np.asarray(mg["theta_mg"])
    err_p = float(np.linalg.norm(th_p - th_t) / np.linalg.norm(th_t))
    err_m = float(np.linalg.norm(th_m - th_t) / np.linalg.norm(th_t))
    ok = err_p < 0.15 and pmg["phi_mean"] < 0.0
    return {
        "synthetic_theta_hat": float(th_p[0]),
        "synthetic_theta_true": float(th_t[0]),
        "synthetic_theta_mg": float(th_m[0]),
        "synthetic_err_pmg": err_p,
        "synthetic_err_mg": err_m,
        "synthetic_phi_mean": float(pmg["phi_mean"]),
        "synthetic_score": float(ok),
    }
