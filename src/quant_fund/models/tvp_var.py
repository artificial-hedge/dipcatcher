"""Time-varying-parameter regression via Kalman filtering.

References
----------
- Primiceri, G.E. (2005). "Time Varying Structural Vector
  Autoregressions and Monetary Policy." *Review of Economic Studies*
  72(3), 821-852.
- Cogley, T. & Sargent, T.J. (2005). "Drifts and Volatilities: Monetary
  Policies and Outcomes in the Post WWII US." *Review of Economic
  Dynamics* 8(2), 262-302.
- Stock, J.H. & Watson, M.W. (1996). "Evidence on Structural
  Instability in Macroeconomic Time Series Relations." *Journal of
  Business & Economic Statistics* 14(1), 11-30.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
State-space model with random-walk coefficients

    y_t = x_t' beta_t + eps_t,   eps ~ N(0, sigma2),
    beta_t = beta_{t-1} + w_t,   w_t ~ N(0, Q),

filtered by the standard Kalman recursion and smoothed by Rauch-Tung-
Striebel backward smoothing. ``Q`` is set by the signal-to-noise
variance ratio ``q = lam * sigma2`` per state (Primiceri-Cogley
"training prior" scale convention) — the state innovation variance that
lets the filter track a mid-sample structural break. The synth shifts
beta at t = T/2; the smoothed TVP path recovers both regimes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def tvp_regression(
    y: FloatArray,
    x: FloatArray,
    lam: float = 0.01,
) -> dict[str, float | FloatArray]:
    """RW-coefficient TVP regression with RTS smoothing.

    Returns the final smoothed coefficients, the in-sample one-step
    forecast error diagnostics, and the path of a break diagnostic.
    """
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.ndim != 1 or xx.ndim != 2 or xx.shape[0] != yy.shape[0]:
        raise ValueError("y must be (n,), x (n,k)")
    n, k = xx.shape
    if n < 30:
        raise ValueError("need n >= 30")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("non-finite inputs")
    if lam <= 0.0:
        raise ValueError("lam must be positive")

    sigma2 = float(np.var(yy - xx @ np.linalg.lstsq(xx, yy, rcond=None)[0]))
    sigma2 = max(sigma2, 1e-10)
    q_mat = lam * sigma2 * np.eye(k)

    beta = np.zeros(k)
    pcov = 10.0 * np.eye(k)
    filt_b = np.zeros((n, k))
    filt_p = np.zeros((n, k, k))
    pred_b = np.zeros((n, k))
    pred_p = np.zeros((n, k, k))
    innov = np.zeros(n)
    for t in range(n):
        bp = beta
        pp = pcov + q_mat
        pred_b[t] = bp
        pred_p[t] = pp
        h = xx[t]
        s = float(h @ pp @ h + sigma2)
        kg = pp @ h / s
        e = yy[t] - h @ bp
        innov[t] = e
        beta = bp + kg * e
        pcov = pp - np.outer(kg, h @ pp)
        filt_b[t] = beta
        filt_p[t] = pcov

    sm_b = np.zeros((n, k))
    sm_b[-1] = filt_b[-1]
    for t in range(n - 2, -1, -1):
        a_gain = filt_p[t] @ np.linalg.solve(pred_p[t + 1], np.eye(k)).T
        sm_b[t] = filt_b[t] + a_gain @ (sm_b[t + 1] - pred_b[t + 1])

    half = n // 2
    mse1 = float(np.mean(innov[10:] ** 2))
    drift = float(np.abs(sm_b[-1] - sm_b[half // 2]).max())
    break_magnitude = float(np.abs(sm_b[3 * n // 4] - sm_b[n // 4]).max())
    return {
        "mse_innovation": mse1,
        "beta_final_0": float(sm_b[-1, 0]),
        "beta_final_1": float(sm_b[-1, 1]),
        "beta_early_0": float(sm_b[n // 4, 0]),
        "beta_early_1": float(sm_b[n // 4, 1]),
        "beta_mid_1": float(sm_b[half, 1]),
        "state_drift": drift,
        "break_magnitude": break_magnitude,
        "loglik_per_obs": float(-0.5 * (np.log(2 * np.pi * sigma2) + 1.0)),
        "_path": sm_b,
    }


def synth_tvp(
    n: int = 400,
    seed: int = 20261231 + 285,
    beta_break: float = 1.2,
) -> dict[str, FloatArray]:
    """Two-regime regression: beta_1 jumps at mid-sample."""
    rng = np.random.default_rng(seed)
    if n < 30:
        raise ValueError("n too small")
    x = np.column_stack([np.ones(n), rng.normal(0.0, 1.0, n)])
    b1 = np.where(np.arange(n) < n // 2, 0.5, 0.5 + beta_break)
    y = x[:, 0] * 0.3 + x[:, 1] * b1 + rng.normal(0.0, 0.3, n)
    return {"y": y, "x": x, "b1_true": b1}


def bench_tvp_var(seed: int = 20261231 + 285) -> dict[str, float]:
    """Wave-49 self-check: smoothed TVP tracks the mid-sample break;
    pre-break and post-break coefficients differ by ~the jump."""
    d = synth_tvp(seed=seed)
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    a = tvp_regression(y, x)
    a2 = tvp_regression(y, x)
    path = np.asarray(a["_path"])
    pre = float(np.mean(path[: path.shape[0] // 3, 1]))
    post = float(np.mean(path[2 * path.shape[0] // 3 :, 1]))
    detects = float(post - pre > 0.6)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(np.array_equal(path, np.asarray(a2["_path"]))),
        "synthetic_pre_beta1": pre,
        "synthetic_post_beta1": post,
        "synthetic_break_jump": post - pre,
        "synthetic_mse": float(a["mse_innovation"]),
        "synthetic_beta0_drift": abs(float(a["beta_early_0"]) - float(a["beta_final_0"])),
    }
