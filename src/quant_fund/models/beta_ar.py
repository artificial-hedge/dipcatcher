"""Rocha-Cribari-Neto beta autoregression for unit-interval series.

References
----------
- Rocha, A.V. & Cribari-Neto, F. (2009). "Beta
  Autoregressive Moving Average Models." *TEST* 18(3),
  529-545.
- Ferrari, S. & Cribari-Neto, F. (2004). "Beta Regression
  for Modelling Rates and Proportions." *Journal of
  Applied Statistics* 31(7), 799-815.
- Casarin, R., Leisen, F., Molina, G. & ter Horst, E.
  (2015). "A Bayesian Beta Markov Random Field
  Calibration of the Term Structure of Implied Risk
  Neutral Densities." *Bayesian Analysis* 10(4), 791-819.
- da-Silva, C.Q., Migon, H.S. & Correia, L.T. (2011).
  "Dynamic Bayesian Beta Models." *Computational
  Statistics & Data Analysis* 55(6), 2074-2089.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
A beta autoregression BARMA(p,q) models a unit-interval
series through a transformed conditional mean: with
``mu_t = g^{-1}(eta_t)``,
``eta_t = alpha + sum phi_k (g(y_{t-k}) - g(mu_{t-k})) +
sum theta_j e_{t-j}``, where g is the logit link and the
error is defined on the link scale, ``e_t = g(y_t) -
g(mu_t)``. Conditional on mu_t and precision nu_t, y_t is
Beta(mu_t nu_t, (1 - mu_t) nu_t). The subtlety that
separates BARMA from naive logit-AR: the AR recursion acts
on the *link-scale error* so the innovation enters
multiplicatively on the beta scale, and the conditional
density must be evaluated at mu_t built recursively — not
at the static marginal mean. We implement BARMA(1,1) with
constant precision nu estimated jointly by conditional
ML/BFGS on the link-scale eta recursion, with a
strictly-inside-(0,1) guard (logit inputs are clamped to
``(eps, 1 - eps)`` since endpoints are outside the
support). ``synth_beta_ar`` simulates the model forward
with phi = 0.65; the bench gates on phi recovery, nu being
sizable (the model is identified by dispersion), and on
one-step-ahead RMSE beating the static-beta marginal mean.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import betaln

FloatArray = NDArray[np.float64]

_EPS = 1e-8


def _logit(x: FloatArray) -> FloatArray:
    xc = np.clip(x, _EPS, 1.0 - _EPS)
    return np.log(xc / (1.0 - xc))


def _expit(eta: FloatArray) -> FloatArray:
    ec = np.clip(eta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-ec))


def _as_series(x: FloatArray, min_len: int = 60) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if np.any(v <= 0.0) or np.any(v >= 1.0):
        raise ValueError("beta series must lie strictly in (0, 1)")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def beta_ar_fit(y: FloatArray) -> dict[str, float]:
    """BARMA(1,1) conditional ML on the link scale."""
    v = _as_series(y)
    n = v.size
    g = _logit(v)

    def nll(theta: FloatArray) -> float:
        alpha, phi, theta1, log_nu = (
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            float(theta[3]),
        )
        if abs(phi) >= 0.999 or abs(theta1) >= 0.999:
            return 1e12
        nu = np.exp(log_nu)
        eta = np.zeros(n)
        eta[0] = alpha / (1.0 - phi) if abs(phi) < 0.999 else alpha
        err = np.zeros(n)
        for t in range(1, n):
            eta[t] = alpha + phi * err[t - 1] + theta1 * (g[t - 1] - eta[t - 1])
            mu_t = float(_expit(np.array([eta[t]]))[0])
            err[t] = g[t] - float(_logit(np.array([mu_t]))[0])
        mu_all = _expit(eta)
        a = mu_all * nu
        b = (1.0 - mu_all) * nu
        ll = (
            -betaln(a, b)
            + (a - 1.0) * np.log(np.clip(v, _EPS, 1.0))
            + (b - 1.0) * np.log(np.clip(1.0 - v, _EPS, 1.0))
        )
        return float(-np.sum(ll[1:]))

    x0 = np.array([0.0, 0.4, 0.0, np.log(30.0)])
    res = minimize(
        nll,
        x0,
        method="BFGS",
        options={"maxiter": 400},
    )
    alpha, phi, theta1, log_nu = (
        float(res.x[0]),
        float(res.x[1]),
        float(res.x[2]),
        float(res.x[3]),
    )
    nu = float(np.exp(log_nu))
    # one-step-ahead fitted means
    eta = np.zeros(n)
    eta[0] = alpha / (1.0 - phi) if abs(phi) < 0.999 else alpha
    err = np.zeros(n)
    for t in range(1, n):
        eta[t] = alpha + phi * err[t - 1] + theta1 * (g[t - 1] - eta[t - 1])
        mu_t = float(_expit(np.array([eta[t]]))[0])
        err[t] = g[t] - float(_logit(np.array([mu_t]))[0])
    mu_hat = _expit(eta)  # fitted conditional means
    out: dict[str, float] = {
        "alpha": alpha,
        "phi": phi,
        "theta": theta1,
        "nu": nu,
        "nll": float(res.fun),
        "converged": float(res.success),
        "rmse_os": float(np.sqrt(np.mean((v[1:] - mu_hat[1:]) ** 2))),
    }
    return out


def synth_beta_ar(
    seed: int = 20261231 + 358,
    n: int = 600,
    alpha: float = -0.3,
    phi: float = 0.65,
    nu: float = 40.0,
) -> FloatArray:
    """SYNTHETIC BARMA(1,0) simulation on the link scale."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eta = np.zeros(n)
    eta[0] = alpha / (1.0 - phi)
    y[0] = float(_expit(np.array([eta[0]]))[0])
    for t in range(1, n):
        mu_prev = float(_expit(np.array([eta[t - 1]]))[0])
        e_prev = float(_logit(np.array([y[t - 1]]))[0]) - float(_logit(np.array([mu_prev]))[0])
        eta[t] = alpha + phi * e_prev
        mu_t = float(_expit(np.array([eta[t]]))[0])
        a = mu_t * nu
        b = (1.0 - mu_t) * nu
        y[t] = float(rng.beta(a, b))
        y[t] = float(np.clip(y[t], _EPS, 1.0 - _EPS))
    return y.astype(np.float64)


def bench_beta_ar(seed: int = 20261231 + 358) -> dict[str, float]:
    y = synth_beta_ar(seed=seed)
    r = beta_ar_fit(y)
    # static-marginal comparator: unconditional mean
    rmse_marg = float(np.sqrt(np.mean((y[1:] - np.mean(y)) ** 2)))
    ok = 0.2 < r["phi"] < 0.95 and r["nu"] > 5.0 and r["rmse_os"] < rmse_marg
    out: dict[str, float] = {
        "synthetic_beta_phi_hat": r["phi"],
        "synthetic_beta_nu_hat": r["nu"],
        "synthetic_beta_rmse_os": r["rmse_os"],
        "synthetic_beta_rmse_marg": rmse_marg,
        "score": 1.0 if ok else 0.0,
    }
    return out
