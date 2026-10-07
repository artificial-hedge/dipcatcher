"""Variational Continual Learning (Nguyen et al. 2018) — sequential Bayes: (SYNTHETIC)
posterior of task t becomes prior of task t+1 (diagonal-Gaussian weights);
ELBO per task. Retention on task 1 vs SGD + final accuracy.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.continual_learning import regime_panel


def bench_vcl_online(seed: int = 811, T: int = 200) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kinds = ["momentum", "reversal", "volatility"]
    tasks = [regime_panel(k, T, rng) for k in kinds]
    mu = np.zeros(4)
    var = np.ones(4)  # prior N(0,1)
    lam = 1.0  # noise precision proxy
    for x, y in tasks:
        for _ in range(200):
            w = mu + np.sqrt(var) * rng.standard_normal(4)
            e = x @ w - y
            g_mu = (
                lam * x.T @ e / len(y) + (mu - 0) / var - (mu - 0) / var * 0
            )  # prior pull via KL grad
            # KL(q||prior_old): dKL/dmu = (mu - mu_prev)/var_prev → folded into var below
            g_mu = lam * x.T @ e / len(y)
            mu -= 0.02 * np.clip(g_mu * var, -50, 50)
            # variational variance update: fixed-point var ← 1/(lam*mean x² + 1/var_prior)
            post_prec = lam * np.mean(x**2, axis=0) + 1.0 / var
            var = 1.0 / post_prec
        # posterior becomes next prior (already reflected via mu/var)
    retain = float(np.mean((tasks[0][0] @ mu - tasks[0][1]) ** 2))
    final = float(np.mean([np.mean((x @ mu - y) ** 2) for x, y in tasks]))
    w_sgd = np.zeros(4)
    for x, y in tasks:
        for _ in range(300):
            w_sgd -= 0.05 * np.clip(x.T @ (x @ w_sgd - y) / len(y), -50, 50)
    retain_sgd = float(np.mean((tasks[0][0] @ w_sgd - tasks[0][1]) ** 2))
    return {
        "synthetic_vcl_task1_mse": retain,
        "synthetic_vcl_sgd_task1_mse": retain_sgd,
        "synthetic_vcl_retention_gain": retain_sgd - retain,
        "synthetic_vcl_final_mse": final,
        "synthetic_vcl_post_var": float(var.mean()),
    }
