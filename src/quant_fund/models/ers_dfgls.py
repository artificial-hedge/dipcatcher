"""Elliott-Rothenberg-Stock (1996) DF-GLS unit-root test.

References
----------
- Elliott, G., Rothenberg, T.J. & Stock, J.H. (1996).
  "Efficient Tests for an Autoregressive Unit Root."
  *Econometrica* 64(4), 813-836.
- Cheung, Y.-W. & Lai, K.S. (1995). "Lag Order and Critical
  Values of a Modified Dickey-Fuller Test." *Oxford Bulletin
  of Economics and Statistics* 57(3), 411-419.
- Ng, S. & Perron, P. (2001). "Lag Length Selection and the
  Construction of Unit Root Tests with Good Size and Power."
  *Econometrica* 69(6), 1519-1554.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
DF-GLS first locally detrends the series at the
near-unit-root alternative ``rho = 1 + c/T`` (c = -7 for the
constant-mean case, c = -13.5 for the trend case): build the
quasi-difference ``w_t = y_t - rho*y_{t-1}``, regress w on
quasi-differenced deterministic components to get GLS
coefficients, and form the detrended ``y^d = y - X*beta``.
Then run the DF regression on y^d *without* deterministic
terms: ``dy^d = a*y^d_{t-1} + sum phi_j dy^d_{t-j} + e`` and
compare the t-statistic on ``a`` to the ERS critical values
(constant: -1.94 at 5% for c=-7; trend: -2.89 for c=-13.5).
Lag order by the Ng-Perron MAIC rule
``MAIC(k) = log sigma2 + 2(k + T0)/T`` with
``T0 = sum phi_j / (1 - sum phi_j)`` — we implement the
standard modified-AIC refinement (Ng-Perron 2001) over
k in 0..k_max. The synth pairs a random walk (accept) with
a stationary AR(0.5) (reject).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _gls_detrend(x: FloatArray, trend: bool) -> FloatArray:
    t = x.size
    c = -13.5 if trend else -7.0
    rho = 1.0 + c / t
    w = np.diff(x) - (rho - 1.0) * x[:-1]
    w = np.concatenate([[x[0]], w])
    if trend:
        z = np.column_stack([np.ones(t), np.arange(1.0, t + 1.0)])
    else:
        z = np.ones((t, 1))
    qd_z = np.diff(z, axis=0) - (rho - 1.0) * z[:-1]
    qd_z = np.vstack([z[:1], qd_z])
    coef, *_ = np.linalg.lstsq(qd_z, w, rcond=None)
    return np.asarray(x - z @ coef)


def _maic_lag(y: FloatArray, k_max: int) -> int:
    """Ng-Perron modified-AIC lag pick on the DF residuals."""
    best_k, best_v = 0, np.inf
    for k in range(k_max + 1):
        dy = np.diff(y)
        n = dy.size - k
        if n < 30:
            break
        # build regressor matrix: level lag + k difference lags
        rows = []
        for i in range(k, dy.size):
            row = [y[i]]
            row += [dy[i - j] for j in range(1, k + 1)]
            rows.append(row)
        xmat = np.asarray(rows)
        target = dy[k:]
        coef, *_ = np.linalg.lstsq(xmat, target, rcond=None)
        resid = target - xmat @ coef
        s2 = float(resid @ resid / resid.size)
        phis = coef[1:]
        tau0 = (
            float(phis.sum() / (1.0 - phis.sum()))
            if phis.size and abs(1 - phis.sum()) > 1e-3
            else 0.0
        )
        v = np.log(s2) + 2.0 * (k + abs(tau0)) / resid.size
        if v < best_v:
            best_v, best_k = v, k
    return best_k


def dfgls_test(
    x: FloatArray,
    trend: bool = False,
    k_max: int | None = None,
) -> dict[str, float]:
    """ERS DF-GLS t-test on the quasi-differenced detrended data."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 80 or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    if np.std(np.diff(xx)) < 1e-12:
        raise ValueError("degenerate")
    yd = _gls_detrend(xx, trend)
    km = k_max if k_max is not None else int(12 * (xx.size / 100) ** 0.25)
    k = _maic_lag(yd, min(km, 8))
    dy = np.diff(yd)
    rows = []
    for i in range(k, dy.size):
        row = [yd[i]] + [dy[i - j] for j in range(1, k + 1)]
        rows.append(row)
    xmat = np.asarray(rows)
    target = dy[k:]
    coef, *_ = np.linalg.lstsq(xmat, target, rcond=None)
    resid = target - xmat @ coef
    dof = resid.size - xmat.shape[1]
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.pinv(xmat.T @ xmat)
    se = float(np.sqrt(cov[0, 0]))
    tstat = float(coef[0] / se)
    crit5 = -2.89 if trend else -1.94
    return {
        "t": tstat,
        "lag": float(k),
        "crit5": crit5,
        "reject_unit_root": float(tstat < crit5),
    }


def synth_dfgls(
    seed: int = 20261231 + 325,
    n: int = 400,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC random walk vs stationary AR(0.5) + level drift."""
    rng = np.random.default_rng(seed)
    rw = np.cumsum(rng.normal(0.0, 1.0, n))
    st = np.zeros(n)
    for t in range(1, n):
        st[t] = 0.5 * st[t - 1] + rng.normal()
    st += 2.0
    return np.asarray(rw), np.asarray(st)


def bench_dfgls(seed: int = 20261231 + 325) -> dict[str, float]:
    """Wave-56 self-check: RW accepted, AR(0.5) rejected."""
    rw, st = synth_dfgls(seed=seed)
    r_rw = dfgls_test(rw)
    r_st = dfgls_test(st)
    ok = r_rw["reject_unit_root"] == 0.0 and r_st["reject_unit_root"] == 1.0
    return {
        "synthetic_t_rw": r_rw["t"],
        "synthetic_t_st": r_st["t"],
        "synthetic_crit5": r_st["crit5"],
        "synthetic_lag_st": r_st["lag"],
        "synthetic_score": float(ok),
    }
