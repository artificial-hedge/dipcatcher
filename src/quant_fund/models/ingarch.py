"""Ferland-Latour-Oraichi Poisson INGARCH(1,1).

References
----------
- Ferland, R., Latour, A. & Oraichi, D. (2006).
  "Integer-Valued GARCH Process." *Journal of Time Series
  Analysis* 27(6), 923-942.
- Heinen, A. (2003). "Modelling Time Series Count Data: An
  Autoregressive Conditional Poisson Model." CORE
  Discussion Paper 2003-62.
- Zhu, F. (2011). "A Negative Binomial Integer-Valued
  GARCH Model." *Journal of Time Series Analysis* 32(1),
  54-67.
- Doukhan, P., Latour, A. & Oraichi, D. (2006). "A Simple
  Integer-Valued Bilinear Time Series Model." *Advances in
  Applied Probability* 38(2), 507-527.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
INGARCH(1,1) drives the Poisson intensity
``lambda_t = omega + alpha * y_{t-1} + beta * lambda_{t-1}``
with y_t ~ Poisson(lambda_t) — the integer-valued analogue
of GARCH where the conditional mean recurses on both the
count and its own lag, giving unconditional mean
``omega / (1 - alpha - beta)`` when ``alpha + beta < 1``.
Estimation is plain conditional Poisson MLE: the score is
``sum y_t log lambda_t - lambda_t`` with lambda built
recursively. The trap we guard against is the weakly-
identified omega/beta corner (flat lambda admits infinitely
many (omega, beta) pairs): the score also reports
``alpha + beta`` (persistence) and the unconditional mean
check, so a degenerate fit at the boundary is caught by
mismatch with the sample mean. Optimization is constrained
Nelder-Mead on transformed parameters (softplus omega,
logit alpha/beta with alpha+beta<1 enforced by the penalty
return). ``synth_ingarch`` simulates the process forward;
the bench gates on persistence recovery, unconditional-
mean agreement, and on the fitted one-step intensity
beating the static Poisson mean on count RMSE.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 60) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if np.any(v < 0):
        raise ValueError("count series must be non-negative")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _lambda_path(y: FloatArray, omega: float, alpha: float, beta: float) -> FloatArray:
    n = y.size
    lam = np.zeros(n)
    lam[0] = max(omega / max(1.0 - alpha - beta, 1e-6), 1e-3)
    for t in range(1, n):
        lam[t] = omega + alpha * y[t - 1] + beta * lam[t - 1]
        lam[t] = max(lam[t], 1e-8)
    return lam


def ingarch_fit(y: FloatArray) -> dict[str, float]:
    """Poisson INGARCH(1,1) conditional MLE."""
    v = _as_series(y)

    def nll(theta: FloatArray) -> float:
        omega = float(np.exp(theta[0]))
        ea, eb = np.exp(theta[1]), np.exp(theta[2])
        alpha = 0.999 * ea / (1.0 + ea + eb)
        beta = 0.999 * eb / (1.0 + ea + eb)
        lam = _lambda_path(v, omega, alpha, beta)
        ll = np.sum(v[1:] * np.log(lam[1:]) - lam[1:] - gammaln(v[1:] + 1.0))
        return float(-ll)

    x0 = np.array([np.log(max(np.mean(v) * 0.1, 1e-2)), -0.5, 0.5])
    res = minimize(
        nll, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8}
    )
    omega = float(np.exp(res.x[0]))
    ea, eb = np.exp(res.x[1]), np.exp(res.x[2])
    alpha = 0.999 * ea / (1.0 + ea + eb)
    beta = 0.999 * eb / (1.0 + ea + eb)
    lam = _lambda_path(v, omega, alpha, beta)
    out: dict[str, float] = {
        "omega": omega,
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(alpha + beta),
        "uncond_mean": float(omega / max(1.0 - alpha - beta, 1e-6)),
        "nll": float(res.fun),
        "rmse_os": float(np.sqrt(np.mean((v[1:] - lam[1:]) ** 2))),
    }
    return out


def synth_ingarch(
    seed: int = 20261231 + 359,
    n: int = 800,
    omega: float = 0.8,
    alpha: float = 0.25,
    beta: float = 0.55,
) -> FloatArray:
    """SYNTHETIC INGARCH(1,1) trajectory."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    lam = np.zeros(n)
    lam[0] = omega / (1.0 - alpha - beta)
    y[0] = float(rng.poisson(lam[0]))
    for t in range(1, n):
        lam[t] = omega + alpha * y[t - 1] + beta * lam[t - 1]
        y[t] = float(rng.poisson(lam[t]))
    return y.astype(np.float64)


def bench_ingarch(seed: int = 20261231 + 359) -> dict[str, float]:
    y = synth_ingarch(seed=seed)
    r = ingarch_fit(y)
    rmse_marg = float(np.sqrt(np.mean((y[1:] - np.mean(y)) ** 2)))
    ok = (
        0.5 < r["persistence"] < 0.95
        and abs(r["uncond_mean"] - float(np.mean(y))) / float(np.mean(y)) < 0.5
        and r["rmse_os"] < rmse_marg
    )
    out: dict[str, float] = {
        "synthetic_ingarch_persist": r["persistence"],
        "synthetic_ingarch_uncond": r["uncond_mean"],
        "synthetic_ingarch_sample_mean": float(np.mean(y)),
        "synthetic_ingarch_rmse_os": r["rmse_os"],
        "synthetic_ingarch_rmse_marg": rmse_marg,
        "score": 1.0 if ok else 0.0,
    }
    return out
