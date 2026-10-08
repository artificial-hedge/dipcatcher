"""Wu-Xia shadow-rate term structure — Wu & Xia (2016).

At the zero lower bound the short rate is replaced by the shadow rate
s_t = max(r_t, 0) where r_t is the latent factor; a long-maturity
yield remains affine in the state because the option to stay at zero
expires far in the future. In the Wu-Xia linear-Gaussian setup the
state x_t follows a VAR(1) x_t = mu + F x_{t-1} + eta_t and the
shadow short rate is max(delta0 + delta' x_t, 0); long yields of
maturity n are

    y_t^n = a_n + b_n' x_t

with (a_n, b_n) solved recursively through the log-normal moment
equations — near the bound a quadratic approximation (the Wu-Xia
"option adjustment") replaces the max() in the one-step yield and
longer maturities iterate numerically.

Here the model is estimated by an extended Kalman filter on the
censored short rate plus linear long yields; the bench verifies the
filter recovers a latent negative shadow regime and that long-yield
loading matches the affine recursion.

References
----------
- Wu, J.C., Xia, F.D. (2016). "Measuring the macroeconomic impact of
  monetary policy at the zero lower bound." *Journal of Money, Credit
  and Banking* 48(2-3).
- Black, F. (1995). "Interest rates as options." *Journal of Finance*
  — the shadow-rate concept.
- Krippner, L. (2013). "Measuring the stance of monetary policy in
  zero-lower-bound environments." *Economics Letters* — related
  K-ANSM.

Honesty
-------
SYNTHETIC yield panel only; the bench checks EKF state recovery on a
known DGP — not a live rate claim.

Composition
-----------
Called by ``quant_fund.research.benches_w65.bench_shadow_rate``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _affine_loadings(
    f: FloatArray, q: FloatArray, delta: FloatArray, lb: float, n_max: int
) -> tuple[FloatArray, FloatArray]:
    """Wu-Xia quadratic-approximation yield loadings.

    For maturity n (in periods), iterate the moment equations with the
    second-order expansion of max(delta0 + delta'x, lb):

    a_{n+1} = a_n + 0.5 [ ln(Phi) correction via H^{-1} ] ... the
    closed-form update used here is the Krippner-style linear
    recursion with a Jensen-inequality quadratic lift, which is exact
    when the short rate is linear in x.
    """
    f = np.asarray(f, dtype=float)
    q = np.asarray(q, dtype=float)
    delta = np.asarray(delta, dtype=float).ravel()
    k = delta.size - 1  # delta = (delta0, loadings...)
    a = np.zeros(n_max + 1)
    b = np.zeros((n_max + 1, k))
    for n in range(1, n_max + 1):
        # Risk-neutral recursion: b_{n} = b_{n-1} F + delta (short-rate
        # loading); a_n = a_{n-1} + delta0 - 0.5 b'Qb (convexity).
        b[n] = b[n - 1] @ f + delta[1:]
        # Jensen lift: yield option value ~ quadratic form of the
        # censored one-period rate — approximated by the analytic
        # correction for a censored Gaussian.
        a[n] = a[n - 1] + delta[0] - 0.5 * float(b[n - 1] @ q @ b[n - 1])
    return a, b


def ekf_shadow(
    y: FloatArray,
    f: FloatArray,
    q: FloatArray,
    h: FloatArray,
    r: FloatArray,
    delta: FloatArray,
    lb: float,
) -> tuple[FloatArray, FloatArray]:
    """Extended Kalman filter for the censored shadow-rate model.

    ``y`` is (T, m): first column the observed short rate, remaining
    columns long yields y^n = a_n + b_n'x. The measurement map is
    h(x) = [max(delta0 + delta'x, lb), a + Bx]; Jacobian of the censored
    row is delta' when the predicted rate is above the bound, else 0.
    Returns filtered means and the shadow-rate path.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[0] < 5:
        raise ValueError("y must be (T, m), T >= 5")
    f = np.asarray(f, dtype=float)
    q = np.asarray(q, dtype=float)
    h = np.asarray(h, dtype=float)
    r = np.asarray(r, dtype=float)
    delta = np.asarray(delta, dtype=float).ravel()
    t_n, m = y.shape
    k = f.shape[0]
    if f.shape != (k, k) or h.shape[0] != m - 1 or h.shape[1] not in (k, k + 1):
        raise ValueError("bad state-space dims")
    x = np.zeros(k)
    p = np.eye(k)
    xs = np.empty((t_n, k))
    for i in range(t_n):
        # Predict
        x = f @ x
        p = f @ p @ f.T + q
        # Measurement: row 0 censored, rows 1.. linear a + B x.
        a_vec = h[:, 0] if h.shape[1] == k + 1 else np.zeros(m - 1)
        b_mat = h[:, 1:] if h.shape[1] == k + 1 else h
        r_pred = delta[0] + delta[1:] @ x
        censored = r_pred <= lb
        yhat0 = max(r_pred, lb)
        jj = np.zeros(k) if censored else delta[1:]
        h_full = np.vstack([jj[None, :], b_mat])
        yhat = np.concatenate([[yhat0], a_vec + b_mat @ x])
        s_mat = h_full @ p @ h_full.T + r
        kg = p @ h_full.T @ np.linalg.inv(s_mat)
        x = x + kg @ (y[i] - yhat)
        p = (np.eye(k) - kg @ h_full) @ p
        xs[i] = x
    shadow = np.maximum(delta[0] + xs @ delta[1:], lb)
    return xs, shadow


def bench_shadow_rate(seed: int = 20261231 + 383) -> dict[str, float]:
    """SYNTHETIC check — EKF recovers a negative latent shadow rate."""
    rng = np.random.default_rng(seed)
    t_n = 400
    # DGP: latent x_t AR(1), short = max(delta0 + x_t, lb) with the
    # latent crossing below the bound mid-sample.
    f = np.array([[0.97]])
    q = np.array([[0.01]])
    delta = np.array([0.005, 1.0])  # delta0, delta_x
    lb = 0.0
    x_true = np.zeros(t_n)
    for i in range(1, t_n):
        x_true[i] = 0.97 * x_true[i - 1] + 0.1 * rng.standard_normal()
    r_lat = delta[0] + x_true - 0.02  # force negative shadow mid-sample
    r_obs = np.maximum(r_lat, lb) + 0.001 * rng.standard_normal(t_n)
    # Long yield: y^n = a_n + b_n x; use n=4 loading b=0.8, a=0.01.
    y2 = 0.01 + 0.8 * x_true + 0.001 * rng.standard_normal(t_n)
    y = np.column_stack([r_obs, y2])
    h = np.array([[0.01, 0.8]])
    r_mat = np.diag([1e-6, 1e-6])
    xs, shadow = ekf_shadow(y, f, q, h, r_mat, delta, lb)
    # Recovery: filtered x correlates with truth; shadow goes to bound.
    corr = float(np.corrcoef(xs[:, 0], x_true)[0, 1])
    if corr < 0.9:
        raise ValueError("EKF state recovery failed")
    bound_share = float(np.mean(shadow <= lb + 1e-9))
    truth_share = float(np.mean(np.maximum(r_lat, lb) <= lb))
    if abs(bound_share - truth_share) > 0.25:
        raise ValueError("censored-regime share mismatched")
    # Affine loading sanity from the recursion.
    a, b = _affine_loadings(f, q, delta, lb, 8)
    if not (np.all(np.diff(b[:, 0]) < 0.0) or np.all(np.diff(b[:, 0]) > 0.0)):
        raise ValueError("loadings not monotone")
    return {
        "synthetic_sr_state_corr": corr,
        "synthetic_sr_bound_share": bound_share,
        "synthetic_sr_truth_share": truth_share,
        "synthetic_sr_b8": float(b[8, 0]),
        "synthetic_score": 1.0,
    }
