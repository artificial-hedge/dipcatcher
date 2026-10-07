"""Cragg two-part hurdle model for semi-continuous outcomes (SYNTHETIC).

A point mass at zero plus a continuous positive part: logit
decides participation, a truncated-normal governs the amount
conditional on trading. Naive OLS conflates the two margins;
the hurdle separates them and Tobit (censored) mis-specifies
zeros as latent negatives.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure parameter recovery on
generated two-part data — never market evidence.

References:
- Cragg, J. G. (1971). Some statistical models for limited
  dependent variables with application to the demand for
  durable goods. *Econometrica* 39, 829-844 — the two-part
  (hurdle) model.
- Cameron, A. C., Trivedi, P. K. (2005). *Microeconometrics:
  Methods and Applications*, ch. 16 — truncated-normal density
  and two-part decomposition tests.
- Mullahy, J. (1986). Specification and testing of some
  modified count data models. *J. Econometrics* 33 — hurdle
  likelihood factorization.
- Jones, A. M. (1989). A double-hurdle model of cigarette
  consumption. *J. Applied Econometrics* 4.

Composition: pure numpy + scipy — logit IRLS for the first
hurdle, truncated-normal MLE on positives; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _logit_fit(w: FloatArray, y_bin: FloatArray, iters: int = 60) -> FloatArray:
    n, k = w.shape
    g = np.linalg.solve(w.T @ w + 1e-6 * np.eye(k), w.T @ y_bin)
    for _ in range(iters):
        xb = np.clip(w @ g, -30, 30)
        p = 1.0 / (1.0 + np.exp(-xb))
        wgt = np.clip(p * (1.0 - p), 1e-8, None)
        grad = w.T @ (y_bin - p)
        h = (w.T * wgt[None, :]) @ w
        try:
            step = np.linalg.solve(h + 1e-8 * np.eye(k), grad)
        except np.linalg.LinAlgError:
            break
        g = g + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return g


def hurdle_fit(y: FloatArray, x: FloatArray) -> dict[str, float]:
    """Cragg hurdle: logit P(y>0|x) + truncated-normal (y|y>0,x).

    ``x`` is (n × k) covariates; intercept added internally.
    Returns both margins plus OLS/Tobit references."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] != yy.size:
        raise ValueError("x rows must match y")
    n, k = xx.shape
    if n < 60 or k < 1 or k > 8:
        raise ValueError("n>=60, k in 1..8")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")
    if np.any(yy < 0):
        raise ValueError("y must be nonnegative (point mass at 0)")
    zeros = int(np.sum(yy == 0.0))
    if zeros < 10 or n - zeros < 30:
        raise ValueError("need >=10 zeros and >=30 positives")
    if np.min(np.std(xx, axis=0)) < 1e-9:
        raise ValueError("each covariate must vary")

    w = np.column_stack([np.ones(n), xx])
    z = (yy > 0).astype(np.float64)

    # first part: participation logit
    gamma = _logit_fit(w, z)
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(w @ gamma, -30, 30)))

    # second part: truncated normal on positives
    pos = yy > 0
    yp, wp = yy[pos], w[pos]

    def nll(theta: FloatArray) -> float:
        beta = theta[:-1]
        sig = float(np.exp(np.clip(theta[-1], -4, 3)))
        mu = wp @ beta
        a = (0.0 - mu) / sig
        log_phi = norm.logpdf((yp - mu) / sig)
        log_1mF = norm.logsf(a)
        return -float(np.sum(log_phi - np.log(sig) - log_1mF))

    b0 = np.linalg.lstsq(wp, yp, rcond=None)[0]
    s0 = float(np.std(yp - wp @ b0))
    res = minimize(
        nll,
        np.concatenate([b0, [np.log(max(s0, 1e-2))]]),
        method="Nelder-Mead",
        options={"maxiter": 3000},
    )
    beta = res.x[:-1]
    sig = float(np.exp(np.clip(res.x[-1], -4, 3)))

    # fitted intensity (unconditional E[y|x])
    mu_pos = w @ beta
    a_all = (0.0 - mu_pos) / sig
    trunc_mean = mu_pos + sig * np.exp(norm.logpdf(a_all) - norm.logsf(a_all))
    ey = p_hat * trunc_mean

    # references: OLS on all; OLS on positives only (selection bias)
    ols = np.linalg.lstsq(w, yy, rcond=None)[0]
    ols_pos = np.linalg.lstsq(wp, yp, rcond=None)[0]

    return {
        "n": float(n),
        "share_zero": float(zeros / n),
        "gamma_x": float(gamma[1]),
        "beta_x": float(beta[1]),
        "sigma": sig,
        "ols_x": float(ols[1]),
        "ols_pos_x": float(ols_pos[1]),
        "ame_x": float(np.mean(p_hat) * beta[1]),
        "fit_cor": float(np.corrcoef(ey, yy)[0, 1]),
    }


def synth_hurdle(
    n: int = 900,
    gamma_x: float = 0.8,
    beta_x: float = 1.5,
    sigma: float = 0.7,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Two-part DGP: participation logit on x, then positive part
    y = xβ + σ ε truncated at zero."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 1))
    p = 1.0 / (1.0 + np.exp(-(0.2 + gamma_x * x[:, 0])))
    trade = rng.uniform(0.0, 1.0, n) < p
    y = np.zeros(n)
    raw = 0.5 + beta_x * x[:, 0] + sigma * rng.normal(0.0, 1.0, n)
    y[trade] = np.clip(raw[trade], 0.05, None)
    return {
        "y": y,
        "x": x,
        "gamma_x": np.array([gamma_x]),
        "beta_x": np.array([beta_x]),
    }


def bench_hurdle(seed: int = 20261231 + 228) -> dict[str, float]:
    """Hurdle self-check: both margins recovered; OLS-on-positives
    bias flagged. All ``synthetic_*``."""
    d = synth_hurdle(seed=seed)
    out = hurdle_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    d0 = synth_hurdle(gamma_x=3.0, beta_x=1.5, seed=seed + 1)
    out0 = hurdle_fit(np.asarray(d0["y"]), np.asarray(d0["x"]))
    out_b = hurdle_fit(np.asarray(d["y"]), np.asarray(d["x"]))

    g = float(out["gamma_x"])
    b = float(out["beta_x"])
    return {
        "synthetic_gamma": g,
        "synthetic_gamma_err": float(abs(g - 0.8)),
        "synthetic_beta": b,
        "synthetic_beta_err": float(abs(b - 1.5)),
        "synthetic_share_zero": float(out["share_zero"]),
        "synthetic_fit_cor": float(out["fit_cor"]),
        "synthetic_pos_bias": float(abs(float(out0["ols_pos_x"]) - 1.5)),
        "synthetic_detects": float(
            abs(g - 0.8) < 0.4 and abs(b - 1.5) < 0.4 and float(out["fit_cor"]) > 0.6
        ),
        "synthetic_determinism": float(
            g == float(out_b["gamma_x"]) and b == float(out_b["beta_x"])
        ),
    }
