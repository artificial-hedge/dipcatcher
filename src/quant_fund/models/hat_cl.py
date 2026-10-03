"""Hard Attention to the Task (Serra et al. 2018) — each task learns
binary feature-gates; previously used gates are locked (embedding
accumulates max) so earlier tasks can't be overwritten.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.continual_learning import regime_panel


def bench_hat_cl(seed: int = 777, T: int = 200) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kinds = ["momentum", "reversal", "volatility"]
    tasks = [regime_panel(k, T, rng) for k in kinds]
    n_feat = 4
    amax = np.zeros(n_feat)  # cumulative max gate
    ws: list[np.ndarray] = []
    for _t, (x, y) in enumerate(tasks):
        w = np.zeros(n_feat)
        gate = np.ones(n_feat) - amax  # free capacity
        for _ in range(300):
            pred = x @ (w * gate)
            g = x.T @ (pred - y) / len(y)
            w -= 0.05 * np.clip(g * gate, -50, 50)
        # lock: gate becomes binary for features actually used
        used = (np.abs(w) > 1e-3).astype(float)
        amax = np.maximum(amax, used)
        ws.append(w * gate)
    retain = float(np.mean((tasks[0][0] @ ws[0] - tasks[0][1]) ** 2))
    w_sgd = np.zeros(n_feat)
    for x, y in tasks:
        for _ in range(300):
            w_sgd -= 0.05 * np.clip(x.T @ (x @ w_sgd - y) / len(y), -50, 50)
    retain_sgd = float(np.mean((tasks[0][0] @ w_sgd - tasks[0][1]) ** 2))
    return {
        "synthetic_hat_task1_mse": retain,
        "synthetic_hat_sgd_task1_mse": retain_sgd,
        "synthetic_hat_retention_gain": retain_sgd - retain,
        "synthetic_hat_locked_features": float(amax.sum()),
    }
