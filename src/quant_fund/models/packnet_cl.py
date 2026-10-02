"""PackNet (Mallya & Lazebnik 2018) — each task gets a disjoint subset of
parameters; once assigned, weights are frozen — zero forgetting by
construction. Retention + final MSE vs plain SGD on the regime stream.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.continual_learning import mse, regime_panel


def _fit_masked(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, iters: int = 300, lr: float = 0.05
) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        e = x @ w - y
        g = x.T @ e / len(y)
        w -= lr * np.clip(g * mask, -50, 50)
    return w


def bench_packnet_cl(seed: int = 761, T: int = 200) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kinds = ["momentum", "reversal", "volatility"]
    tasks = [regime_panel(k, T, rng) for k in kinds]
    n_feat = 4
    # disjoint masks: feature j belongs to task j % n_tasks
    masks = [np.array([1.0 if j % 3 == t else 0.0 for j in range(n_feat)]) for t in range(3)]
    w_all = np.zeros(n_feat)
    for t, (x, y) in enumerate(tasks):
        w_t = _fit_masked(x, y, masks[t])
        w_all = w_all + w_t * masks[t]
    retain = mse(type("M", (), {"predict": lambda s, xx: xx @ w_all})(), tasks[0][0], tasks[0][1])
    final = np.mean(
        [mse(type("M", (), {"predict": lambda s, xx: xx @ w_all})(), x, y) for x, y in tasks]
    )
    # plain SGD baseline
    w_sgd = np.zeros(n_feat)
    for x, y in tasks:
        for _ in range(300):
            g = x.T @ (x @ w_sgd - y) / len(y)
            w_sgd -= 0.05 * np.clip(g, -50, 50)
    retain_sgd = float(np.mean((tasks[0][0] @ w_sgd - tasks[0][1]) ** 2))
    return {
        "synthetic_pn_task1_mse": retain,
        "synthetic_pn_sgd_task1_mse": retain_sgd,
        "synthetic_pn_retention_gain": retain_sgd - retain,
        "synthetic_pn_final_mse": float(final),
    }
