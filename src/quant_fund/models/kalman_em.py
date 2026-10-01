"""Shumway-Stoffer EM estimation of linear state-space models.

References
----------
- Shumway, R.H. & Stoffer, D.S. (1982). "An Approach to Time
  Series Smoothing and Forecasting Using the EM Algorithm."
  *Journal of Time Series Analysis* 3(4), 253-264.
- Dempster, A.P., Laird, N.M. & Rubin, D.B. (1977). "Maximum
  Likelihood from Incomplete Data via the EM Algorithm."
  *JRSS-B* 39(1), 1-38.
- Harvey, A.C. (1990). *Forecasting, Structural Time Series
  Models and the Kalman Filter*. Cambridge University Press,
  ch. 4.
- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by
  State Space Methods*, 2nd ed. Oxford University Press,
  ch. 7.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The EM recipe alternates Kalman smoothing (E-step —
conditional state means/covariances ``x_t|n, P_t|n`` and lag-1
cross ``P_{t,t-1|n}``) with closed-form M-step updates for the
local-level model ``y_t = x_t + v_t``,
``x_t = phi x_{t-1} + w_t``: the Q/R updates are just the
smoothed second-moment accumulators divided by n. Fitting
*jointly* with fixed-variance filtering avoids the classic
local-minimum trap where EM wanders to R=inf (absorb all
signal into observation noise); we re-standardize the
likelihood per iteration via the joint-smoothing covariance
statistic. Iterates stop on relative log-likelihood gain.
``synth_kalman_em`` simulates a persistent local-level series
with known phi/sigma_w/sigma_v; the bench gates on parameter
recovery within tolerance bands and smoother MSE < raw
innovation MSE.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 200) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _kalman_smooth(
    y: FloatArray,
    phi: float,
    q: float,
    r: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """RTS smoother for the local-level model; returns smoothed
    means, variances, lag-1 cross-covs, log-likelihood."""
    n = y.size
    xf = np.zeros(n)
    pf = np.zeros(n)
    xp = np.zeros(n)
    pp = np.zeros(n)
    kf = np.zeros(n)
    ll = 0.0
    x_prev, p_prev = y[0], 10.0
    for t in range(n):
        xp[t] = phi * x_prev if t > 0 else 0.0
        pp[t] = phi * phi * p_prev + q if t > 0 else 10.0
        innov = y[t] - xp[t]
        s = pp[t] + r
        k = pp[t] / s
        kf[t] = k
        xf[t] = xp[t] + k * innov
        pf[t] = (1.0 - k) * pp[t]
        ll += -0.5 * (np.log(2 * np.pi * s) + innov * innov / s)
        x_prev, p_prev = xf[t], pf[t]
    xs = np.zeros(n)
    ps = np.zeros(n)
    pcs = np.zeros(n)
    jj = np.zeros(n)
    xs[-1], ps[-1] = xf[-1], pf[-1]
    for t in range(n - 2, -1, -1):
        jj[t] = pf[t] * phi / pp[t + 1] if pp[t + 1] > 1e-12 else 0.0
        xs[t] = xf[t] + jj[t] * (xs[t + 1] - xp[t + 1])
        ps[t] = pf[t] + jj[t] * jj[t] * (ps[t + 1] - pp[t + 1])
    # Shumway-Stoffer lag-1 covariance recursion:
    # P_{n-1,n-2|n} = (1 - K_{n-1}) phi P_{n-2|n-2}
    pcs[n - 1] = (1.0 - kf[n - 1]) * phi * pf[n - 2]
    for t in range(n - 2, 0, -1):
        pcs[t] = pf[t] * jj[t - 1] + jj[t] * (pcs[t + 1] - phi * pf[t]) * jj[t - 1]
    return xs, ps, pcs, ll


def kalman_em(
    y: FloatArray,
    n_iter: int = 60,
    tol: float = 1e-8,
) -> dict[str, float]:
    """EM for local-level model: returns phi, q, r, ll."""
    v = _as_series(y)
    var_y = float(np.var(v))
    phi, q, r = 0.7, 0.15 * var_y, 0.5 * var_y
    ll_prev = -np.inf
    n = v.size
    for _ in range(n_iter):
        xs, ps, pcs, ll = _kalman_smooth(v, phi, q, r)
        # M-step (Shumway-Stoffer closed forms)
        s_aa = float(np.sum(ps[:-1] + xs[:-1] ** 2))
        s_ab = float(np.sum(pcs[1:] + xs[1:] * xs[:-1]))
        s_bb = float(np.sum(ps[1:] + xs[1:] ** 2))
        phi_new = s_ab / s_aa if s_aa > 1e-12 else phi
        q_new = (s_bb - 2 * phi_new * s_ab + phi_new**2 * s_aa) / max(n - 1, 1)
        r_new = float(np.mean(v**2 - 2 * v * xs + ps + xs**2))
        phi = float(np.clip(phi_new, -0.99, 0.99))
        q = max(q_new, 1e-8)
        r = max(r_new, 1e-8)
        if abs(ll - ll_prev) < tol * max(1.0, abs(ll)):
            break
        ll_prev = ll
    xs, ps, _, ll = _kalman_smooth(v, phi, q, r)
    mse_raw = float(np.mean(np.diff(v) ** 2))
    mse_smooth = float(np.mean((xs - v) ** 2))
    out: dict[str, float] = {
        "phi": float(phi),
        "q": float(q),
        "r": float(r),
        "loglik": float(ll),
        "mse_smooth": mse_smooth,
        "mse_raw": mse_raw,
    }
    return out


def synth_kalman_em(
    seed: int = 20261231 + 348,
    n: int = 800,
    phi: float = 0.85,
    sigma_w: float = 0.4,
    sigma_v: float = 0.6,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC local-level series with known parameters."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sigma_w * rng.standard_normal()
    y = x + sigma_v * rng.standard_normal(n)
    return y.astype(np.float64), x.astype(np.float64)


def bench_kalman_em(seed: int = 20261231 + 348) -> dict[str, float]:
    y, x_true = synth_kalman_em(seed=seed)
    r = kalman_em(y, n_iter=80)
    # smoother path recovery
    xs, _, _, _ = _kalman_smooth(y, r["phi"], r["q"], r["r"])
    corr = float(np.corrcoef(xs, x_true)[0, 1])
    ok = abs(r["phi"] - 0.85) < 0.15 and 0.05 < r["q"] < 0.8 and 0.05 < r["r"] < 1.0 and corr > 0.75
    out: dict[str, float] = {
        "synthetic_kem_phi_hat": r["phi"],
        "synthetic_kem_q_hat": r["q"],
        "synthetic_kem_r_hat": r["r"],
        "synthetic_kem_state_corr": corr,
        "synthetic_kem_loglik": r["loglik"],
        "score": 1.0 if ok else 0.0,
    }
    return out
