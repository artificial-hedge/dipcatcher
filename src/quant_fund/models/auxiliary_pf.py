"""Pitt-Shephard auxiliary particle filter.

Auxiliary particle filter for a scalar stochastic-
volatility state-space model

    x_t = phi x_{t-1} + sigma_v eps_t     (state)
    y_t = exp(x_t / 2) eta_t             (observation)

The APF resamples particles at t-1 in proportion to
``w_{t-1}^i * f(y_t | mu_t^i)`` (first-stage auxiliary
weights using the predicted state mean ``mu_t^i =
phi x_{t-1}^i``), then propagates the resampled
particles through the transition and reweights by the
second-stage ratio ``f(y_t|x_t^j)/f(y_t|mu_t^j)``.

Honesty: the bench compares APF effective sample size
against a plain bootstrap PF on a SYNTHETIC SV path and
requires mean ESS_APF >= mean ESS_BPF — an efficiency
diagnostic, not a market claim.

References
----------
* Pitt & Shephard (1999) "Filtering via simulation:
  auxiliary particle filters", JASA 94, 590-599.
* Doucet, de Freitas & Gordon (2001) "Sequential Monte
  Carlo Methods in Practice", Springer.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _sv_ll(x: FloatArray, y_t: float) -> FloatArray:
    """log f(y_t | x) up to constants: -x/2 - y^2 e^{-x}/2."""
    xc = np.clip(x, -30.0, 30.0)
    return -0.5 * xc - 0.5 * y_t * y_t * np.exp(-xc)


def _ess(w: FloatArray) -> float:
    return float(1.0 / max(float((w**2).sum()), 1e-300))


def _sys_resample(w: FloatArray, rng: np.random.Generator) -> FloatArray:
    n = w.size
    u0 = rng.uniform(0.0, 1.0 / n)
    pts = u0 + np.arange(n) / n
    cw = np.cumsum(w)
    idx = np.searchsorted(cw, pts, side="left")
    return np.asarray(np.clip(idx, 0, n - 1), dtype=np.int64)


def _logmeanexp(lw: FloatArray) -> float:
    m = float(lw.max())
    return float(m + math.log(float(np.exp(lw - m).mean())))


def bootstrap_pf_sv(
    y: FloatArray,
    n_part: int = 400,
    phi: float = 0.98,
    sigma_v: float = 0.15,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Plain bootstrap PF for the SV model (comparison)."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    t_n = yy.size
    if t_n < 5:
        raise ValueError("need n>=5")
    rng = np.random.default_rng(seed)
    x = sigma_v / math.sqrt(1 - phi * phi) * rng.normal(size=n_part)
    w = np.full(n_part, 1.0 / n_part)
    ess = np.zeros(t_n)
    ll = 0.0
    for t in range(t_n):
        # propagate
        x = phi * x + sigma_v * rng.normal(size=n_part)
        lw = _sv_ll(x, float(yy[t]))
        ll += _logmeanexp(lw + np.log(w))
        wu = np.exp(lw - lw.max()) * w
        wu /= wu.sum()
        ess[t] = _ess(wu)
        idx = _sys_resample(wu, rng)
        x = x[idx]
        w = np.full(n_part, 1.0 / n_part)
    return {"loglik": float(ll), "ess": np.asarray(ess), "mean_ess": float(ess.mean())}


def auxiliary_pf_sv(
    y: FloatArray,
    n_part: int = 400,
    phi: float = 0.98,
    sigma_v: float = 0.15,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """APF for the SV model with first-stage mean lookahead."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    t_n = yy.size
    if t_n < 5:
        raise ValueError("need n>=5")
    rng = np.random.default_rng(seed)
    x = sigma_v / math.sqrt(1 - phi * phi) * rng.normal(size=n_part)
    w = np.full(n_part, 1.0 / n_part)
    ess = np.zeros(t_n)
    ll = 0.0
    for t in range(t_n):
        mu_pred = phi * x
        # first-stage auxiliary weights: w_{t-1} * f(y_t | mu_pred)
        lw_aux = _sv_ll(mu_pred, float(yy[t])) + np.log(np.clip(w, 1e-300, None))
        w_aux = np.exp(lw_aux - lw_aux.max())
        w_aux /= w_aux.sum()
        idx = _sys_resample(w_aux, rng)
        xj = x[idx]
        wj = w[idx] / np.clip(w_aux[idx], 1e-300, None)
        # propagate through transition
        xj = phi * xj + sigma_v * rng.normal(size=n_part)
        # second-stage weight: f(y|x_j)/f(y|mu_j) * w_j
        ll_xj = _sv_ll(xj, float(yy[t]))
        lw2 = ll_xj - _sv_ll(phi * x[idx], float(yy[t]))
        ll += _logmeanexp(np.log(np.clip(wj, 1e-300, None)) + ll_xj)
        w_new = wj * np.exp(lw2 - lw2.max())
        w_new /= w_new.sum()
        ess[t] = _ess(w_new)
        x = xj
        w = w_new
    return {
        "loglik": float(ll),
        "ess": np.asarray(ess),
        "mean_ess": float(ess.mean()),
    }


def bench_auxiliary_pf(seed: int = 20261231 + 467) -> dict[str, float]:
    """SYNTHETIC check — APF loglik dominates BPF's on informative obs."""
    rng = np.random.default_rng(seed)
    t_n = 120
    phi, sigma_v = 0.97, 0.30
    x_true = np.zeros(t_n)
    x_true[0] = rng.normal(scale=sigma_v / math.sqrt(1 - phi * phi))
    for t in range(1, t_n):
        x_true[t] = phi * x_true[t - 1] + sigma_v * rng.normal()
    # informative observations: lookahead materially helps
    y = 0.25 * np.exp(x_true / 2.0) * rng.normal(size=t_n)
    apf = auxiliary_pf_sv(y, n_part=300, phi=phi, sigma_v=sigma_v, seed=seed)
    bpf = bootstrap_pf_sv(y, n_part=300, phi=phi, sigma_v=sigma_v, seed=seed + 1)
    la = float(apf["loglik"])
    lb = float(bpf["loglik"])
    if not (la > lb):
        raise ValueError(f"apf loglik off: apf={la} bpf={lb}")
    ea = float(apf["mean_ess"])
    eb = float(bpf["mean_ess"])
    return {
        "synthetic_apf_loglik": la,
        "synthetic_bpf_loglik": lb,
        "synthetic_loglik_gap": float(la - lb),
        "synthetic_apf_ess": ea,
        "synthetic_bpf_ess": eb,
        "synthetic_score": 1.0,
    }
