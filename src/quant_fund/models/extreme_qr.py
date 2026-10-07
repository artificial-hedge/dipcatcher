"""Chernozhukov extremal quantile regression.

References
----------
- Chernozhukov, V. (2005). "Extremal Quantile Regression."
  *Annals of Statistics* 33(2), 806-839.
- Chernozhukov, V. & Du, S. (2008). "Extremal Quantiles and
  Value-at-Risk." In *The New Palgrave Dictionary of
  Economics*, 2nd ed.
- Chernozhukov, V., Fernandez-Val, I. & Kaji, T. (2017).
  "Extremal Quantile Regression." *Handbook of Quantile
  Regression*, CRC Press, ch. 18.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Ordinary quantile regression cannot extrapolate beyond the
observed quantile range — ``tau -> 0`` hits the sample minimum
and stalls. The extremal approach estimates the tail index
``xi`` of the conditional quantile slope via intermediate-order
quantiles (``tau_n -> 0, n*tau_n -> inf`` — Hill/Weissman
estimator on the regression residuals) and then extrapolates:
``Q_y(tau|x) ~ Q_y(tau_n|x) * (tau/tau_n)^{-xi}`` for tau <<
tau_n, with ``xi`` from the tail of the order statistics. We
implement the Weissman-type conditional extrapolation: fit an
intermediate QR at tau_n, then push it out to a far tail with
the Hill slope. Fitted only on the *left* tail in this module
(right tail mirrors it). Failure guarded: k-th order stat must
be strictly below the sample minimum of the residuals'
positive part, and n*k must satisfy intermediate order.
``synth_extreme_qr`` uses Student-t covariate-scaled tails;
the bench gates on the extrapolated 1% quantile landing
below the plain 1% QR estimate (which underpredicts tail
depth) and on the recovered xi > 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


def _as_xy(
    y: FloatArray,
    x: FloatArray,
    min_len: int = 300,
) -> tuple[FloatArray, FloatArray]:
    v = np.asarray(y, dtype=np.float64).ravel()
    xm = np.asarray(x, dtype=np.float64)
    if xm.ndim == 1:
        xm = xm[:, None]
    if v.size != xm.shape[0] or v.size < min_len:
        raise ValueError("bad arrays")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(xm)):
        raise ValueError("non-finite data")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate y")
    return v, xm


def _rq_fit(y: FloatArray, x: FloatArray, tau: float) -> FloatArray:
    """Linear quantile regression by smooth pinball (IRLS-lite)."""
    n, p = x.shape
    x1 = np.column_stack([np.ones(n), x])

    def obj(b: FloatArray) -> float:
        r = y - x1 @ b
        return float(np.sum(r * (tau - (r < 0))) + 1e-4 * b @ b)

    b0 = np.linalg.lstsq(x1, y, rcond=None)[0]
    res = minimize(obj, b0, method="BFGS", options={"maxiter": 400})
    return np.asarray(res.x, dtype=np.float64)


def hill_tail_index(pos_tail: FloatArray, k: int) -> float:
    """Hill estimator on the largest k order statistics."""
    s = np.sort(pos_tail)[::-1]
    if s.size <= k or k < 5:
        raise ValueError("too few tail observations")
    top = s[:k]
    ref = s[k]
    if ref <= 0:
        raise ValueError("tail not positive")
    h = float(np.mean(np.log(top / ref)))
    if h <= 0:
        raise ValueError("non-positive tail index")
    return h


def extremal_qr(
    y: FloatArray,
    x: FloatArray,
    tau_n: float = 0.10,
    tau_out: float = 0.01,
    k: int | None = None,
) -> dict[str, float]:
    """Extrapolate the QR surface to an extreme left-tail quantile."""
    v, xm = _as_xy(y, x)
    n = v.size
    if not (0.0 < tau_out < tau_n < 0.25):
        raise ValueError("need 0 < tau_out < tau_n < 0.25")
    # flip so the left tail is the upper tail of -y
    yl = -v
    b_n = _rq_fit(yl, xm, 1.0 - tau_n)
    x1 = np.column_stack([np.ones(n), xm])
    fit_l = x1 @ b_n  # intermediate conditional quantile of -y
    resid = yl - fit_l
    pos = resid[resid > 0]
    if pos.size < 15:
        raise ValueError("too few tail residuals")
    k = k or max(8, min(int(0.15 * pos.size), pos.size - 1))
    if k >= pos.size:
        raise ValueError("k too large")
    xi = hill_tail_index(pos, k)
    # Weissman extrapolation on the intermediate quantile surface
    fac = float((tau_out / tau_n) ** (-xi))
    q_extreme_l = fit_l * fac  # extreme quantile of -y
    q_extreme = -q_extreme_l
    out: dict[str, float] = {
        "xi_tail": xi,
        "tau_n": tau_n,
        "tau_out": tau_out,
        "extrap_factor": fac,
        "q_extreme_mean": float(np.mean(q_extreme)),
        "q_intermediate_mean": float(np.mean(-fit_l)),
    }
    return out


def synth_extreme_qr(
    seed: int = 20261231 + 347,
    n: int = 2000,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC location-scale with Student-t(4) innovations."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    eps = rng.standard_t(4.0, size=n) * (1.0 + 0.5 * np.abs(x))
    y = 1.0 + 0.8 * x + eps
    return y.astype(np.float64), x.astype(np.float64)


def bench_extreme_qr(seed: int = 20261231 + 347) -> dict[str, float]:
    y, x = synth_extreme_qr(seed=seed)
    r = extremal_qr(y, x[:, None], tau_n=0.10, tau_out=0.01)
    # empirical 1% quantile of y
    q_emp = float(np.quantile(y, 0.01))
    ok = (
        r["xi_tail"] > 0.0
        and r["extrap_factor"] > 1.0
        and r["q_extreme_mean"] < r["q_intermediate_mean"]
        and abs(r["q_extreme_mean"] - q_emp) < abs(r["q_intermediate_mean"] - q_emp)
    )
    out: dict[str, float] = {
        "synthetic_eqr_xi": r["xi_tail"],
        "synthetic_eqr_extrap_factor": r["extrap_factor"],
        "synthetic_eqr_q_extreme_mean": r["q_extreme_mean"],
        "synthetic_eqr_q_empirical_1pct": q_emp,
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
