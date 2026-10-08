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
        # snapshot the task's prior: posterior of task t-1 — the KL
        # pull must reference THIS prior, not the drifting current q.
        mu_pr, var_pr = mu.copy(), var.copy()
        for _ in range(200):
            w = mu + np.sqrt(var) * rng.standard_normal(4)
            e = x @ w - y
            # KL(q||prior): d/dmu = (mu - mu_pr)/var_pr — the prior pull
            # was previously computed and then overwritten, leaving
            # plain preconditioned SGD instead of the VCL update.
            g_mu = lam * x.T @ e / len(y) + (mu - mu_pr) / var_pr
            mu -= 0.02 * np.clip(g_mu * var_pr, -50, 50)
            # variational variance update: var ← 1/(lam*mean x² + 1/var_pr)
            post_prec = lam * np.mean(x**2, axis=0) + 1.0 / var_pr
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
