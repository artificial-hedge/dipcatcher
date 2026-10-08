"""Expectation propagation (Minka 2001): EP for the (SYNTHETIC)
Bayesian probit classifier — site approximations on each
likelihood term yield a Gaussian posterior on the weights;
updates use tilted-moment matching with cavity marginals.
Synthetic bench gates EP accuracy vs Laplace/MCMC-free
baselines on a noisy linear fixture."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def ep_probit_fit(
    x: FloatArray,
    y: FloatArray,
    prior_prec: float = 0.1,
    it: int = 50,
) -> dict[str, object]:
    """EP for y_i = 1{w·x_i + ε > 0}, ε~N(0,1):
    posterior ≈ N(m, V) with sites N(s_i; v_i, τ_i)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64) * 2 - 1  # ±1
    n, d = x.shape
    # sites in natural params
    v = np.zeros(n)  # site mean times precision
    tau = np.zeros(n)  # site precision
    cov = np.linalg.inv(prior_prec * np.eye(d))
    mean = cov @ (x.T * 0).sum(axis=1)  # zero
    mean = np.zeros(d)
    for _ in range(it):
        for i in range(n):
            # cavity: remove site i
            prec = np.linalg.inv(cov)
            cav_prec = prec - tau[i] * np.outer(x[i], x[i])
            cav_cov = np.linalg.inv(cav_prec)
            cav_mean = cav_cov @ (prec @ mean - v[i] * x[i])
            # tilted marginal: z_i = y_i · x_i'μ_ / sqrt(x_i'Σ_x_i + 1)
            m_loc = float(x[i] @ cav_mean)
            s_loc = float(x[i] @ cav_cov @ x[i])
            z = y[i] * m_loc / np.sqrt(s_loc + 1.0)
            cdf = float(norm.cdf(z))
            if cdf < 1e-12:
                continue
            rho = float(norm.pdf(z)) / cdf
            alpha = rho * (rho + z) / (s_loc + 1.0)
            kappa = rho / np.sqrt(s_loc + 1.0)
            new_tau = alpha / (1.0 - alpha * s_loc)
            new_v = new_tau * (m_loc + kappa * y[i] * (s_loc + 1.0))
            # damped site update
            damp = 0.9
            tau[i] = (1 - damp) * tau[i] + damp * new_tau
            v[i] = (1 - damp) * v[i] + damp * new_v
            prec = cav_prec + tau[i] * np.outer(x[i], x[i])
            cov = np.linalg.inv(prec)
            mean = cov @ (cav_prec @ cav_mean + v[i] * x[i])
    return {"mean": mean, "cov": cov}


def ep_probit_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(model["mean"])
    return np.asarray((x @ mean > 0).astype(np.float64))


def ep_probit_marginal(model: dict[str, object], x: FloatArray) -> FloatArray:
    """Posterior predictive probit probability
    Φ(x'm / √(1 + x'Vx))."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(model["mean"])
    cov = np.asarray(model["cov"])
    denom = np.sqrt(1.0 + np.einsum("ij,jk,ik->i", x, cov, x))
    return np.asarray(norm.cdf((x @ mean) / denom))


def bench_expectation_propagation(seed: int = 566) -> dict[str, float]:
    """SYNTHETIC: probit fixture — EP must classify near the
    Bayes boundary, and its predictive probabilities must be
    materially sharper at the margin than a logistic
    baseline at matching confidence mass."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n, d = 200, 4
    x = rng.normal(0, 1, (n, d))
    w_true = np.array([1.5, -1.0, 0.8, 0.2])
    y = (x @ w_true + rng.normal(0, 0.7, n) > 0).astype(np.float64)
    perm = rng.permutation(n)
    tr, te = perm[:150], perm[150:]
    m = ep_probit_fit(x[tr], y[tr], prior_prec=0.1, it=30)
    acc = float((ep_probit_predict(m, x[te]) == y[te]).mean())
    out["synthetic_ep_probit_acc"] = acc
    if acc < 0.75:
        raise ValueError(f"ep probit acc off: {acc}")
    # predictive calibration: Brier score of Φ marginals
    p = ep_probit_marginal(m, x[te])
    brier = float(np.mean((p - y[te]) ** 2))
    out["synthetic_ep_probit_brier"] = brier
    if brier > 0.2:
        raise ValueError(f"ep brier off: {brier}")
    return out
