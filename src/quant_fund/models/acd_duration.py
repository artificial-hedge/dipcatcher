"""Engle-Russell autoregressive conditional duration (ACD).

References
----------
- Engle, R.F. & Russell, J.R. (1998). "Autoregressive Conditional
  Duration: A New Model for Irregularly Spaced Transaction Data."
  *Econometrica* 66(5), 1127-1162.
- Bauwens, L. & Giot, P. (2000). "The Logarithmic ACD Model: An
  Application to the Bid-Ask Quote Process of Three NYSE Stocks."
  *Annales d'Economie et de Statistique* 60, 117-149.
- Engle, R.F. (2000). "The Econometrics of Ultra-High-Frequency Data."
  *Econometrica* 68(1), 1-22.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
ACD(1,1) with standard-unit innovation ``eps_t``:

    x_t = psi_t * eps_t,  E[eps] = 1,
    psi_t = omega + alpha * x_{t-1} + beta * psi_{t-1}.

Parameters are estimated by quasi-MLE under the exponential reference
density ``log L = -sum(log psi_t + x_t / psi_t)`` via a Nelder-Mead
simplex on ``(omega, alpha, beta)``; the stationarity diagnostic
``alpha + beta`` and the standardized-residual autocorrelation report
fit quality. The synth draws the process with ``alpha = 0.15``,
``beta = 0.75``; the QMLE recovers persistence ``> .8`` and leaves
near-white standardized residuals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt

FloatArray = NDArray[np.float64]


def _acf1(x: FloatArray) -> float:
    xc = x - x.mean()
    d = float(xc @ xc)
    if d <= 1e-18:
        return 0.0
    return float(xc[1:] @ xc[:-1] / d)


def acd_fit(
    durations: FloatArray,
    max_iter: int = 400,
) -> dict[str, float]:
    """QMLE fit of ACD(1,1) on positive durations."""
    x = np.asarray(durations, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] < 60:
        raise ValueError("need >= 60 durations")
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("durations must be positive and finite")
    n = x.shape[0]
    xbar = float(np.mean(x))

    def nll(par: FloatArray) -> float:
        om, al, be = float(par[0]), float(par[1]), float(par[2])
        if om <= 0 or al < 0 or be < 0 or al + be >= 0.999:
            return 1e12
        psi = np.empty(n)
        psi[0] = xbar
        for t in range(1, n):
            psi[t] = om + al * x[t - 1] + be * psi[t - 1]
            if psi[t] <= 1e-12 or not np.isfinite(psi[t]):
                return 1e12
        return float(np.sum(np.log(psi) + x / psi))

    def bfgs() -> FloatArray:
        best = np.array([0.1 * xbar, 0.1, 0.8])
        best_val = nll(best)
        for start in ([0.05 * xbar, 0.05, 0.9], [0.2 * xbar, 0.2, 0.6], [0.02 * xbar, 0.02, 0.95]):
            res = _opt.minimize(
                nll,
                np.array(start),
                method="Nelder-Mead",
                options={"maxiter": max_iter, "xatol": 1e-10, "fatol": 1e-10},
            )
            if res.fun < best_val:
                best_val = float(res.fun)
                best = np.asarray(res.x, dtype=np.float64)
        return best

    par = bfgs()
    om, al, be = float(par[0]), float(par[1]), float(par[2])
    psi = np.empty(n)
    psi[0] = xbar
    for t in range(1, n):
        psi[t] = om + al * x[t - 1] + be * psi[t - 1]
    eps = x / psi
    return {
        "omega": om,
        "alpha": al,
        "beta": be,
        "persistence": al + be,
        "mean_duration": xbar,
        "eps_acf1": _acf1(eps),
        "eps_mean": float(np.mean(eps)),
        "nll": nll(par),
    }


def synth_acd(
    n: int = 2000,
    seed: int = 20261231 + 287,
    omega: float = 0.05,
    alpha: float = 0.15,
    beta: float = 0.75,
) -> dict[str, FloatArray]:
    """Simulated ACD(1,1) with exponential innovations."""
    rng = np.random.default_rng(seed)
    if n < 60:
        raise ValueError("n too small")
    eps = rng.exponential(1.0, n)
    x = np.empty(n)
    psi = np.empty(n)
    psi[0] = omega / (1.0 - alpha - beta)
    x[0] = psi[0] * eps[0]
    for t in range(1, n):
        psi[t] = omega + alpha * x[t - 1] + beta * psi[t - 1]
        x[t] = psi[t] * eps[t]
    return {"durations": x, "psi_true": psi}


def bench_acd_duration(seed: int = 20261231 + 287) -> dict[str, float]:
    """Wave-50 self-check: QMLE recovers high persistence and leaves
    near-white standardized residuals."""
    d = synth_acd(seed=seed)
    x = np.asarray(d["durations"])
    a = acd_fit(x)
    a2 = acd_fit(x)
    detects = float(a["persistence"] > 0.7 and abs(float(a["eps_acf1"])) < 0.15)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == a2),
        "synthetic_persistence": a["persistence"],
        "synthetic_alpha": a["alpha"],
        "synthetic_beta": a["beta"],
        "synthetic_eps_acf1": float(a["eps_acf1"]),
        "synthetic_eps_mean": a["eps_mean"],
    }
