"""Phillips-Perron (1988) nonparametric unit-root tests.

References
----------
- Phillips, P.C.B. & Perron, P. (1988). "Testing for a Unit
  Root in Time Series Regression." *Biometrika* 75(2),
  335-346.
- Phillips, P.C.B. (1987). "Time Series Regression with a
  Unit Root." *Econometrica* 55(2), 277-301.
- Schwert, G.W. (1989). "Tests for Unit Roots: A Monte Carlo
  Investigation." *Journal of Business & Economic
  Statistics* 7(2), 147-159.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
PP keeps the simple DF regression
``dy = a + pi * y_{t-1} + e`` (constant optional trend) but
corrects the statistics nonparametrically for serial
correlation and heteroskedasticity in ``e`` via the Newey-West
long-run variance ``s_lrv`` and its ratio
``lam = s_lrv / s_e``:

    Z(pi)  = T * pi_hat - (T^3/(24 s_y)) * (s_lrv^2 - s_e^2)
             [Ori-Hausmann correction on the coefficient]
    Z(t)   = (s_e / s_lrv) * t_pi
             - (T / (4 s_lrv)) * (s_lrv^2 - s_e^2)
             * (T^{-2} sum y_{t-1}^2)^{-1/2} * sigma_adj,

implemented in their standard textbook form (Hamilton 1994,
Table 17.2). Bandwidth ``l4 = trunc(4 (T/100)^0.25)`` per the
KPSS/Schwert convention. The synth contrasts a random walk
(accept) with AR(0.5) plus MA(1) innovations — where naive
DF would be oversized and the PP correction is the point of
the exercise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _nw_lrv(u: FloatArray, lag: int) -> float:
    t = u.size
    v = float(u @ u / t)
    for k in range(1, lag + 1):
        g = float(u[k:] @ u[:-k] / t)
        v += 2.0 * (1.0 - k / (lag + 1)) * g
    return v


def pp_test(
    x: FloatArray,
    trend: bool = False,
    lags: int | None = None,
) -> dict[str, float]:
    """Phillips-Perron Z(pi) and Z(t) statistics."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 80 or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    if np.std(np.diff(xx)) < 1e-12:
        raise ValueError("degenerate")
    dy = np.diff(xx)
    lag_y = xx[:-1]
    n = dy.size
    if trend:
        z = np.column_stack([np.ones(n), lag_y, np.arange(1.5, n + 1.5)])
    else:
        z = np.column_stack([np.ones(n), lag_y])
    coef, *_ = np.linalg.lstsq(z, dy, rcond=None)
    u = dy - z @ coef
    s_e2 = float(u @ u / (n - z.shape[1]))
    lag = lags if lags is not None else int(4 * (n / 100) ** 0.25)
    lag = max(lag, 1)
    s_lrv2 = _nw_lrv(u, lag)
    cov = s_e2 * np.linalg.pinv(z.T @ z)
    se_pi = float(np.sqrt(cov[1, 1]))
    pi = float(coef[1])
    t_pi = pi / se_pi
    syy2 = float(np.sum(lag_y * lag_y))
    lam = s_lrv2 / s_e2
    lam_fac = np.sqrt(s_e2 / s_lrv2)
    # Z(t): Hamilton 17.6.6 constant-case form
    if syy2 <= 1e-20:
        raise ValueError("degenerate regressors")
    z_t = lam_fac * t_pi - 0.5 * (s_lrv2 - s_e2) / np.sqrt(s_lrv2) * (n / np.sqrt(syy2)) * se_pi
    # Z(pi): Hamilton 17.6.11
    z_pi = n * pi - (n * n * se_pi * se_pi) / (2.0 * s_e2) * (s_lrv2 - s_e2)
    # DF 5% critical values apply to Z-stats (constant case)
    crit_pi, crit_t = (-29.3 if trend else -20.7), (-3.45 if trend else -2.89)
    return {
        "z_pi": float(z_pi),
        "z_t": float(z_t),
        "crit_pi5": crit_pi,
        "crit_t5": crit_t,
        "reject_unit_root": float(z_pi < crit_pi and z_t < crit_t),
        "lags": float(lag),
        "lam": float(lam),
    }


def synth_pp(
    seed: int = 20261231 + 327,
    n: int = 400,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC RW vs AR(0.5)+MA(1) — serial-correlation stress."""
    rng = np.random.default_rng(seed)
    e1 = rng.normal(0.0, 1.0, n + 1)
    rw = np.cumsum(e1[1:] + 0.3 * e1[:-1])
    e2 = rng.normal(0.0, 1.0, n + 1)
    st = np.zeros(n)
    for t in range(1, n):
        st[t] = 0.5 * st[t - 1] + e2[t] + 0.4 * e2[t - 1]
    return np.asarray(rw), np.asarray(st)


def bench_phillips_perron(
    seed: int = 20261231 + 327,
) -> dict[str, float]:
    """Wave-56 self-check: RW accepted, AR(0.5)+MA rejected."""
    rw, st = synth_pp(seed=seed)
    r_rw = pp_test(rw)
    r_st = pp_test(st)
    ok = r_rw["reject_unit_root"] == 0.0 and r_st["reject_unit_root"] == 1.0
    return {
        "synthetic_z_pi_rw": r_rw["z_pi"],
        "synthetic_z_t_rw": r_rw["z_t"],
        "synthetic_z_pi_st": r_st["z_pi"],
        "synthetic_z_t_st": r_st["z_t"],
        "synthetic_score": float(ok),
    }
