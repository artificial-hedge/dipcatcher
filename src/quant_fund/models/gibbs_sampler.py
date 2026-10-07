"""Gibbs sampling canon: conjugate normal-mean hierarchical Gibbs (SYNTHETIC)
(mu | tau, x-bar ; tau | IG prior) and Bayesian linear regression
(beta | sigma^2 normal, sigma^2 | IG). ``bench_gibbs_sampler`` gates
posterior-mean recovery on both fixtures plus simple Monte-Carlo
coverage of the credible intervals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def gibbs_normal_mean(
    y: FloatArray,
    tau0: float = 1.0,
    a0: float = 1.0,
    b0: float = 1.0,
    it: int = 4000,
    burn: int = 500,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y ~ N(mu, 1/tau) iid; mu ~ N(0, 1/tau0), tau ~ Ga(a0,b0)."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2 or it <= burn:
        raise ValueError("bad inputs")
    rng = np.random.default_rng(seed)
    n = y.size
    ybar = float(y.mean())
    mu = ybar
    tau = 1.0
    mus: list[float] = []
    taus: list[float] = []
    for t in range(it):
        prec = tau0 + n * tau
        mu = rng.normal((n * tau * ybar) / prec, 1.0 / np.sqrt(prec))
        tau = rng.gamma(a0 + 0.5 * n, 1.0 / (b0 + 0.5 * np.sum((y - mu) ** 2)))
        if t >= burn:
            mus.append(mu)
            taus.append(tau)
    return {
        "mu": np.asarray(mus, dtype=np.float64),
        "tau": np.asarray(taus, dtype=np.float64),
    }


def gibbs_lm(
    x: FloatArray,
    y: FloatArray,
    ridge: float = 1.0,
    a0: float = 1.0,
    b0: float = 1.0,
    it: int = 3000,
    burn: int = 500,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y ~ N(x beta, s2 I); beta ~ N(0, I/ridge*s2), s2 ~ IG(a0,b0)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    if it <= burn or n <= d:
        raise ValueError("bad inputs")
    rng = np.random.default_rng(seed)
    xtx = x.T @ x
    xty = x.T @ y
    s2 = 1.0
    beta = np.zeros(d)
    out_b: list[FloatArray] = []
    out_s: list[float] = []
    eye = np.eye(d)
    for t in range(it):
        prec = xtx + ridge * eye
        cov = s2 * np.linalg.inv(prec)
        mean = np.linalg.solve(prec, xty)
        beta = mean + np.linalg.cholesky(cov) @ rng.standard_normal(d)
        resid = y - x @ beta
        shape = a0 + 0.5 * (n + d)
        scale = b0 + 0.5 * (resid @ resid + ridge * (beta @ beta))
        s2 = 1.0 / rng.gamma(shape, 1.0 / scale)
        if t >= burn:
            out_b.append(beta.copy())
            out_s.append(s2)
    return {
        "beta": np.stack(out_b),
        "sigma2": np.asarray(out_s, dtype=np.float64),
    }


def bench_gibbs_sampler(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # (a) normal mean: truth mu=1.5, sd=0.7
    y = rng.normal(1.5, 0.7, 120)
    g = gibbs_normal_mean(y, it=3000, burn=500, seed=seed)
    mu_s = g["mu"]
    lo, hi = np.quantile(mu_s, [0.05, 0.95])
    # (b) linear regression: truth beta=[1,-2,0.5], sigma=0.4
    n, d = 200, 3
    x = np.column_stack([np.ones(n), rng.standard_normal((n, d - 1))])
    b_true = np.array([1.0, -2.0, 0.5])
    yr = x @ b_true + 0.4 * rng.standard_normal(n)
    g2 = gibbs_lm(x, yr, it=3000, burn=500, seed=seed + 1)
    bh = g2["beta"].mean(axis=0)
    s2h = float(np.sqrt(g2["sigma2"].mean()))
    return {
        "synthetic_gibbs_mu_err": float(abs(mu_s.mean() - 1.5)),
        "synthetic_gibbs_mu_ci_cover": float(lo <= 1.5 <= hi),
        "synthetic_gibbs_mu_ci_width": float(hi - lo),
        "synthetic_gibbs_beta_err": float(np.linalg.norm(bh - b_true)),
        "synthetic_gibbs_sigma_err": float(abs(s2h - 0.4)),
    }
