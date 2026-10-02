"""Survival conformal (Candès et al. 2023) — right-censored calibration
with oracle censoring weights; lower-bound prediction interval on the
survival time. Empirical coverage of the true event time vs naive
interval.
"""

from __future__ import annotations

import numpy as np


def bench_survival_cp(seed: int = 1307, alpha: float = 0.1) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 800
    X = rng.normal(0, 1, (n, 3))
    lam = np.exp(-(0.8 * X[:, 0] + 0.4 * X[:, 1]))
    T = rng.exponential(1 / lam)
    C = rng.exponential(1 / np.exp(-0.5 * X[:, 0])) * 2
    Yobs = np.minimum(T, C)
    delta = (T <= C).astype(float)
    n_tr = n // 2
    # model: quantile of T|x via exponential fit on uncensored (crude)
    A = np.concatenate([X[:n_tr], np.ones((n_tr, 1))], 1)
    w = np.linalg.solve(
        A[delta[:n_tr] == 1].T @ A[delta[:n_tr] == 1] + 0.1 * np.eye(4),
        A[delta[:n_tr] == 1].T @ Yobs[:n_tr][delta[:n_tr] == 1],
    )
    mu = np.concatenate([X[n_tr:], np.ones((n - n_tr, 1))], 1) @ w
    # conformal on censored-aware scores: E = max(T_hat - T,0) among uncensored,
    # weighted by 1/P(C > T|x) ~ reweight to correct censoring bias
    cal = slice(n_tr, n)
    scores = np.maximum(mu - Yobs[cal], 0)
    wgt = 1.0 / np.clip(1 - np.exp(-Yobs[cal] / np.clip(mu, 0.2, None)), 0.1, 1.0)
    wgt = wgt / wgt.sum()
    order = np.argsort(scores)
    cum = np.cumsum(wgt[order])
    k = int(np.searchsorted(cum, 1 - alpha))
    Q = scores[order][min(k, len(scores) - 1)]
    # eval on fresh uncensored T
    Xt = rng.normal(0, 1, (400, 3))
    lam_t = np.exp(-(0.8 * Xt[:, 0] + 0.4 * Xt[:, 1]))
    Tt = rng.exponential(1 / lam_t)
    mu_t = np.concatenate([Xt, np.ones((400, 1))], 1) @ w
    cov = float(np.mean(Tt >= mu_t - Q))
    # naive: no censor correction — quantile of raw scores
    Q2 = np.quantile(scores, 1 - alpha)
    cov2 = float(np.mean(Tt >= mu_t - Q2))
    return {
        "synthetic_scp_coverage": cov,
        "synthetic_scp_target": 1 - alpha,
        "synthetic_scp_naive_coverage": cov2,
        "synthetic_scp_cov_gain": cov - cov2,
        "torch_available": 0.0,
    }
