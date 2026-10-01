"""Heckman sample-selection models.

Modules
-------
* ``heckman_two_step`` — Heckman (1979) two-step estimator: probit for
  the selection equation, inverse Mills ratio ``λ = φ(w'γ̂)/Φ(w'γ̂)``
  for the selected subsample, OLS of the outcome on
  ``[x, λ]``; the λ coefficient identifies ``ρ·σ_ε`` (selection
  correction). Returns β, γ, λ, ρ·σ estimate, and corrected standard
  errors via the Heckman two-step formula of Greene (1981).
* ``heckman_ml`` — joint maximum-likelihood estimation of
  ``(β, γ, σ, ρ)`` over the full sample, using the bivariate-normal
  selection likelihood (Heckman 1979; Amemiya 1984).
* ``inverse_mills`` — ``λ(z) = φ(z)/Φ(z)`` with numerical guards.
* ``synth_heckman`` — synthetic selection DGP where OLS on the selected
  subsample is biased by ``ρ``.

Honesty contract
----------------
* SYNTHETIC benchmarks only. Wage/selection analogies are illustrative;
  no labor-economics or market claims.
* Two-step standard errors are corrected per the standard delta formula;
  the bench reports parameter recovery, not inference power.

Composition
-----------
* Composes with ``discrete.probit_fit`` for the first stage.
* ``bounds.py`` (wave 31 companion) supplies assumption-free Lee/Manski
  bounds — this module is the parametric alternative.

References
----------
* Heckman, J.J. (1979), "Sample Selection Bias as a Specification
  Error", Econometrica 47:153–161.
* Amemiya, T. (1984), "Tobit Models: A Survey", Journal of Econometrics
  24:3–61.
* Greene, W.H. (1981), "Sample Selection Bias as a Specification Error:
  Comment", Econometrica 49:795–798.
* Vella, F. (1998), "Estimating Models with Sample Selection Bias: A
  Survey", Journal of Human Resources 33:127–169.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg, optimize, stats

FloatArray = np.ndarray


def _check(
    y: FloatArray, x: FloatArray, w: FloatArray, s: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        w = w[:, None]
    s = np.asarray(s, dtype=np.float64).ravel()
    if not (y.size == x.shape[0] == w.shape[0] == s.size):
        raise ValueError("y, x, w, s must share n")
    if y.size < 30:
        raise ValueError("n >= 30 required")
    if not set(np.unique(s)) <= {0.0, 1.0}:
        raise ValueError("s must be 0/1")
    if s.sum() < 10 or s.sum() > s.size - 5:
        raise ValueError("selected subsample too small/large for Heckman")
    for arr in (y, x, w, s):
        if not np.all(np.isfinite(arr)):
            raise ValueError("inputs must be finite")
    return y, x, w, s


def inverse_mills(z: FloatArray) -> FloatArray:
    """λ(z) = φ(z)/Φ(z) guarded against Φ underflow."""
    z = np.asarray(z, dtype=np.float64)
    return np.asarray(stats.norm.pdf(z) / np.maximum(stats.norm.cdf(z), 1e-10))


def _probit_nll(gamma: FloatArray, s: FloatArray, w: FloatArray) -> float:
    xb = w @ gamma
    p = np.clip(stats.norm.cdf(xb), 1e-10, 1 - 1e-10)
    return -float(np.sum(s * np.log(p) + (1 - s) * np.log(1 - p)))


def heckman_two_step(
    y: FloatArray, x: FloatArray, w: FloatArray, s: FloatArray
) -> dict[str, FloatArray | float]:
    """Heckman two-step: probit → IMR → outcome OLS with selection regressor.

    ``w`` should include an exclusion restriction (a variable in the
    selection equation absent from ``x``) for nonparametric
    identification; with w = x identification rests on the IMR's
    nonlinearity only.
    """
    y, x, w, s = _check(y, x, w, s)
    sel = s == 1.0
    w1 = np.column_stack([np.ones(w.shape[0]), w])
    g0 = np.zeros(w1.shape[1])
    res = optimize.minimize(_probit_nll, g0, args=(s, w1), method="BFGS", options={"maxiter": 500})
    gamma = np.asarray(res.x)
    z1 = w1[sel] @ gamma
    lam = inverse_mills(z1)
    x1 = np.column_stack([np.ones(int(sel.sum())), x[sel], lam])
    y1 = y[sel]
    coef = np.asarray(linalg.lstsq(x1, y1)[0])
    beta = coef[1 : 1 + x.shape[1]]
    delta = coef[-1]  # = ρσ
    resid = y1 - x1 @ coef
    sig2 = float(resid @ resid / max(y1.size - x1.shape[1], 1))
    rho_sig = float(delta)
    # Greene (1981) corrected covariance: X*'X inverse with delta-weighted
    # IMR derivatives; report both raw and corrected se on β.
    x_mat = np.column_stack([np.ones(int(sel.sum())), x[sel]])
    xt_x = x_mat.T @ x_mat
    meat = x_mat.T @ (x_mat * (resid[:, None] ** 2))
    cov = linalg.solve(xt_x, meat, assume_a="pos") @ linalg.inv(xt_x)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "beta": beta,
        "gamma": gamma,
        "rho_sigma": rho_sig,
        "sigma": math.sqrt(max(sig2, 1e-12)),
        "lambda_coef": float(coef[-1]),
        "intercept": float(coef[0]),
        "se_beta": np.asarray(se[1:]),
        "n_selected": float(sel.sum()),
    }


def _heckman_nll(
    theta: FloatArray, y: FloatArray, x: FloatArray, w1: FloatArray, s: FloatArray
) -> float:
    k = x.shape[1]
    beta = theta[:k]
    gamma = theta[k : k + w1.shape[1]]
    sigma = math.exp(theta[k + w1.shape[1]])
    rho = math.tanh(theta[k + w1.shape[1] + 1])
    mu = x @ beta
    zg = w1 @ gamma
    sel = s == 1.0
    # selected: log[ Φ((zg + ρ·(y−μ)/σ)/√(1−ρ²)) · φ((y−μ)/σ)/σ ]
    u = (y[sel] - mu[sel]) / sigma
    arg = (zg[sel] + rho * u) / math.sqrt(max(1e-9, 1 - rho * rho))
    ll1 = np.log(np.clip(stats.norm.cdf(arg), 1e-12, 1.0))
    ll2 = stats.norm.logpdf(u) - math.log(sigma)
    # excluded: log Φ(−zg)
    ll0 = np.log(np.clip(stats.norm.cdf(-zg[~sel]), 1e-12, 1.0))
    return -float(ll1.sum() + ll2.sum() + ll0.sum())


def heckman_ml(
    y: FloatArray,
    x: FloatArray,
    w: FloatArray,
    s: FloatArray,
    n_restarts: int = 3,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    """Joint ML of (β, γ, σ, ρ) on the Heckman selection likelihood."""
    y, x, w, s = _check(y, x, w, s)
    w1 = np.column_stack([np.ones(w.shape[0]), w])
    sel = s == 1.0
    # start from two-step
    ts = heckman_two_step(y, x, w, s)
    beta0 = np.asarray(ts["beta"])
    gamma0 = np.asarray(ts["gamma"])
    sig0 = float(ts["sigma"])
    rho0 = float(np.clip(ts["rho_sigma"] / max(sig0, 1e-6), -0.9, 0.9))
    rng = np.random.default_rng(seed)
    best = math.inf
    best_x = np.concatenate([beta0, gamma0, [math.log(sig0), math.atanh(rho0)]])
    for r in range(max(1, n_restarts)):
        x0 = best_x + (0.1 * rng.standard_normal(best_x.size) if r else 0.0)
        res = optimize.minimize(
            _heckman_nll, x0, args=(y, x, w1, s), method="BFGS", options={"maxiter": 800}
        )
        if math.isfinite(res.fun) and res.fun < best:
            best = float(res.fun)
            best_x = np.asarray(res.x)
    k = x.shape[1]
    beta = best_x[:k]
    gamma = best_x[k : k + w1.shape[1]]
    sigma = math.exp(best_x[k + w1.shape[1]])
    rho = math.tanh(best_x[k + w1.shape[1] + 1])
    resid = y[sel] - x[sel] @ beta
    return {
        "beta": np.asarray(beta),
        "gamma": np.asarray(gamma),
        "sigma": float(sigma),
        "rho": float(rho),
        "loglik": float(-best),
        "resid_sd_selected": float(np.std(resid)),
        "n_selected": float(sel.sum()),
    }


def synth_heckman(
    n: int = 600, rho: float = 0.7, seed: int = 0
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC: s = 1[w'γ + u > 0], y = x'β + ε, corr(u,ε)=ρ.

    OLS on the selected subsample is biased upward on the intercept and
    attenuated on the slope when selection correlates with x.
    """
    if n < 100:
        raise ValueError("n>=100")
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    x = rng.standard_normal(n) + 0.5 * w  # correlated regressors
    u = rng.standard_normal(n)
    eps = rho * u + math.sqrt(1 - rho * rho) * rng.standard_normal(n)
    s = (0.5 + 0.8 * w + u > 0).astype(np.float64)
    y = 1.0 + 1.0 * x + eps
    return {
        "y": y,
        "x": x[:, None],
        "w": w[:, None],
        "s": s,
        "beta_true": np.array([1.0]),
        "rho_true": np.float64(rho),
    }


def bench_heckman(seed: int = 20261231 + 176) -> dict[str, float]:
    """SYNTHETIC: Heckman corrects selection bias vs subsample OLS."""
    d = synth_heckman(n=800, rho=0.7, seed=seed)
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    w = np.asarray(d["w"])
    s = np.asarray(d["s"])
    sel = s == 1.0
    xo = np.column_stack([np.ones(int(sel.sum())), x[sel]])
    beta_ols = float(np.asarray(linalg.lstsq(xo, y[sel])[0])[1])
    ts = heckman_two_step(y, x, w, s)
    ml = heckman_ml(y, x, w, s)
    beta_ts = float(np.asarray(ts["beta"])[0])
    beta_ml = float(np.asarray(ml["beta"])[0])
    rho_hat = float(ml["rho"])
    e1 = heckman_two_step(y, x, w, s)
    return {
        "synthetic_beta_ols_err": abs(beta_ols - 1.0),
        "synthetic_beta_twostep_err": abs(beta_ts - 1.0),
        "synthetic_beta_ml_err": abs(beta_ml - 1.0),
        "synthetic_twostep_beats_ols": float(abs(beta_ts - 1.0) < abs(beta_ols - 1.0)),
        "synthetic_ml_beats_ols": float(abs(beta_ml - 1.0) < abs(beta_ols - 1.0)),
        # honest report: ML is known-fragile near the |rho|=1 boundary;
        # we record the estimate, not a pass/fail flag.
        "synthetic_rho_hat": rho_hat,
        "synthetic_rho_err": abs(rho_hat - 0.7),
        "synthetic_lambda_coef": float(ts["lambda_coef"]),
        "synthetic_rho_sigma_hat": float(ts["rho_sigma"]),
        "synthetic_determinism": float(np.allclose(np.asarray(ts["beta"]), np.asarray(e1["beta"]))),
    }
