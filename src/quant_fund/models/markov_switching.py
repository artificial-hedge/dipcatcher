"""Markov-switching regime models: Hamilton filter + EM.

Regime-dependent dynamics where the latent state ``s_t ∈ {0,…,K−1}``
follows a first-order Markov chain with transition matrix ``P`` and
``y_t | s_t = j ~ N(μ_j, σ_j²)``. The **Hamilton (1989) filter** gives
exact filtered probabilities ``P(s_t=j | y_{1:t})`` and the log-
likelihood; the **Kim smoother** gives ``P(s_t=j | y_{1:T})``; EM
re-estimates ``(μ, σ, P)`` — the standard toolkit for bull/bear
regime detection on returns-like series.

References
----------
- Hamilton, J. D. (1989). *A new approach to the economic analysis of
  nonstationary time series and the business cycle.* Econometrica
  57(2), 357–384.
- Hamilton, J. D. (1994). *Time Series Analysis.* Princeton — ch. 22,
  filter/smoothing recursions.
- Kim, C.-J. & Nelson, C. R. (1999). *State-Space Models with Regime
  Switching.* MIT Press.

Honesty contract
----------------
``synth_*`` helpers and ``bench_*`` emit SYNTHETIC correctness checks
only — never market evidence; regime labels are arbitrary up to
permutation (reported via the high-μ ordering).

Composition notes
-----------------
numpy/scipy only; deterministic ``np.random.default_rng(seed)``;
fail-closed ``ValueError`` on non-stochastic ``P`` or degenerate
likelihood; EM is run to a fixed iteration budget with a monotone-
likelihood sanity check.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

FloatArray = np.ndarray

__all__ = [
    "bench_markov_switching",
    "hamilton_filter",
    "ms_em_fit",
    "ms_smoother",
    "regime_forecast",
    "synth_markov",
]


def _check_series(y: FloatArray) -> FloatArray:
    a = np.asarray(y, dtype=np.float64)
    if a.ndim != 1 or a.size < 60:
        raise ValueError("need a 1-D series with at least 60 observations")
    if not np.isfinite(a).all():
        raise ValueError("series must be finite")
    return a


def _check_params(
    mu: FloatArray, sigma: FloatArray, p: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    k = mu.size
    if mu.ndim != 1 or sigma.shape != mu.shape or p.shape != (k, k):
        raise ValueError("mu, sigma length-K; P is K×K")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("P must be row-stochastic")
    if k < 2 or k > 6:
        raise ValueError("K in 2..6")
    return mu, sigma, p


def _emission(y: FloatArray, mu: FloatArray, sigma: FloatArray) -> FloatArray:
    """``eta[t,j] = N(y_t; mu_j, sigma_j^2)``."""
    return np.asarray(stats.norm.pdf(y[:, None], loc=mu[None, :], scale=sigma[None, :]))


def hamilton_filter(
    y: FloatArray,
    mu: FloatArray,
    sigma: FloatArray,
    p: FloatArray,
    init: FloatArray | None = None,
) -> dict[str, FloatArray | float]:
    """Hamilton filter: filtered state probabilities + log-likelihood.

    ``xi_t = P(s_t | y_{1:t})`` via ``xi_t ∝ eta_t ⊙ (Pᵀ xi_{t−1})``
    normalized each step; the normalizer accumulates the likelihood.
    """
    y = _check_series(y)
    mu, sigma, p = _check_params(mu, sigma, p)
    k = mu.size
    eta = np.maximum(_emission(y, mu, sigma), 1e-300)
    xi = np.empty((y.size, k))
    if init is None:
        # ergodic initial distribution of P
        w, v = np.linalg.eig(p.T)
        pi0 = np.real(v[:, np.argmax(np.real(w))])
        pi0 = np.abs(pi0) / np.abs(pi0).sum()
    else:
        pi0 = _check_measure(init)
    pred = pi0
    ll = 0.0
    for t in range(y.size):
        pred = p.T @ xi[t - 1] if t else pi0
        xi_t = pred * eta[t]
        s = xi_t.sum()
        if s <= 0:
            raise ValueError("likelihood underflow — degenerate parameters")
        xi[t] = xi_t / s
        ll += math.log(s)
    return {"filtered": xi, "loglik": ll, "eta": eta, "mu": mu, "sigma": sigma, "p": p}


def ms_smoother(filt: dict[str, FloatArray | float]) -> FloatArray:
    """Kim's backward smoother: ``P(s_t | y_{1:T})``.

    ``xi_{t|T} = xi_{t|t} ⊙ [P · (xi_{t+1|T} / xi_{t+1|t})]`` computed
    backward from the last filtered state.
    """
    filtered = np.asarray(filt["filtered"])
    p = np.asarray(filt["p"])
    t_n, k = filtered.shape
    sm = np.empty_like(filtered)
    sm[-1] = filtered[-1]
    for t in range(t_n - 2, -1, -1):
        pred = p.T @ filtered[t]
        ratio = sm[t + 1] / np.maximum(pred, 1e-300)
        sm[t] = filtered[t] * (p @ ratio)
        s = sm[t].sum()
        sm[t] = sm[t] / s if s > 0 else filtered[t]
    return np.asarray(np.clip(sm, 0.0, 1.0))


def ms_em_fit(
    y: FloatArray,
    n_states: int = 2,
    n_iter: int = 30,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    """EM fit of the Gaussian Markov-switching model.

    E-step: filter + smoother. M-step: weighted means/variances for
    ``(mu, sigma)`` and row-normalized expected transition counts for
    ``P`` (expected transitions approximated via smoothed marginals —
    the standard Hamilton simplification)."""
    y = _check_series(y)
    rng = np.random.default_rng(seed)
    mu = np.quantile(y, np.linspace(0.2, 0.8, n_states)) + rng.normal(0, 0.01, n_states)
    sigma = np.full(n_states, float(y.std()))
    p = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
    np.fill_diagonal(p, 0.95)
    p = p / p.sum(axis=1, keepdims=True)
    ll_prev = -math.inf
    for _ in range(n_iter):
        filt = hamilton_filter(y, mu, sigma, p)
        sm = ms_smoother(filt)
        ll = float(filt["loglik"])
        if ll < ll_prev - 1e-6:
            break  # monotone-likelihood guard — degenerate step
        ll_prev = ll
        w = sm
        mu = (w * y[:, None]).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)
        var = (w * (y[:, None] - mu[None, :]) ** 2).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)
        sigma = np.sqrt(np.maximum(var, 1e-8))
        # transition counts from filtered pairs
        xi = np.asarray(filt["filtered"])
        eta = np.asarray(filt["eta"])
        counts = np.zeros((n_states, n_states))
        for t in range(1, y.size):
            joint = p * xi[t - 1][:, None] * eta[t][None, :]
            counts += joint / max(joint.sum(), 1e-300)
        p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
        p = np.clip(p, 1e-4, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
    filt = hamilton_filter(y, mu, sigma, p)
    sm = ms_smoother(filt)
    order = np.argsort(mu)
    return {
        "mu": mu[order],
        "sigma": sigma[order],
        "p": p[order][:, order],
        "filtered": np.asarray(filt["filtered"])[:, order],
        "smoothed": sm[:, order],
        "loglik": float(filt["loglik"]),
        "n_states": float(n_states),
    }


def regime_forecast(filt: dict[str, FloatArray | float], h: int = 5) -> dict[str, FloatArray]:
    """``h``-step regime probabilities and predictive mean/variance from
    the last filtered state via ``P^h``."""
    if h < 1:
        raise ValueError("h >= 1")
    xi_t = np.asarray(filt["filtered"])[-1]
    p = np.asarray(filt["p"])
    mu = np.asarray(filt["mu"])
    sigma = np.asarray(filt["sigma"])
    xi_h = np.linalg.matrix_power(p.T, h) @ xi_t
    mean = float(xi_h @ mu)
    var = float(xi_h @ (sigma**2 + mu**2) - mean**2)
    return {"xi_h": xi_h, "mean": np.array([mean]), "var": np.array([var])}


def _check_measure(w: FloatArray) -> FloatArray:
    a = np.asarray(w, dtype=np.float64)
    s = float(a.sum())
    if a.ndim != 1 or s <= 0 or np.any(a < 0):
        raise ValueError("initial distribution invalid")
    return a / s


def synth_markov(
    n: int = 600,
    seed: int = 0,
    p_stay: float = 0.95,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC two-regime series: low-vol state μ=-0.3/σ=0.4,
    high-vol state μ=+0.8/σ=1.2, persistence ``p_stay``."""
    rng = np.random.default_rng(seed)
    mu = np.array([-0.3, 0.8])
    sigma = np.array([0.4, 1.2])
    p = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < p_stay else 1 - s[t - 1]
    y = mu[s] + sigma[s] * rng.standard_normal(n)
    return {"y": y, "states": s.astype(np.float64), "mu": mu, "sigma": sigma, "p": p}


def bench_markov_switching(seed: int = 20261231 + 184) -> dict[str, float]:
    """SYNTHETIC: EM recovers regime params; smoothed states track truth."""
    d = synth_markov(n=800, seed=seed)
    y = np.asarray(d["y"])
    true_states = np.asarray(d["states"])
    fit = ms_em_fit(y, n_states=2, n_iter=25, seed=seed)
    mu_hat = np.asarray(fit["mu"])
    sigma_hat = np.asarray(fit["sigma"])
    sm = np.asarray(fit["smoothed"])
    # classify each t into the high-μ state by smoothed prob
    pred_hi = (sm[:, 1] > 0.5).astype(np.float64)
    agree = float(np.mean(pred_hi == true_states))
    filt = hamilton_filter(y, mu_hat, sigma_hat, np.asarray(fit["p"]))
    fc = regime_forecast(filt, h=10)
    fit2 = ms_em_fit(y, n_states=2, n_iter=25, seed=seed)
    return {
        "synthetic_mu_err": float(np.abs(np.sort(mu_hat) - np.sort(np.asarray(d["mu"]))).max()),
        "synthetic_sigma_err": float(
            np.abs(np.sort(sigma_hat) - np.sort(np.asarray(d["sigma"]))).max()
        ),
        "synthetic_regime_accuracy": agree,
        "synthetic_persist_hat": float(np.asarray(fit["p"])[0, 0] + np.asarray(fit["p"])[1, 1])
        / 2.0,
        "synthetic_loglik": float(fit["loglik"]),
        "synthetic_forecast_mean": float(np.asarray(fc["mean"])[0]),
        "synthetic_forecast_var": float(np.asarray(fc["var"])[0]),
        "synthetic_determinism": float(np.allclose(np.asarray(fit["mu"]), np.asarray(fit2["mu"]))),
    }
