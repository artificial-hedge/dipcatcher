"""Learning without Forgetting (Li & Hoiem 2016) — after each task the (SYNTHETIC)
model's soft outputs on the new data become distillation targets; loss =
new-task MSE + lam * KL to old predictions.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.continual_learning import regime_panel


def bench_lwf_cl(seed: int = 763, T: int = 200, lam: float = 2.0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kinds = ["momentum", "reversal", "volatility"]
    tasks = [regime_panel(k, T, rng) for k in kinds]
    w = np.zeros(4)
    w_prev: np.ndarray | None = None
    lr = 0.05
    for x, y in tasks:
        for _ in range(300):
            e = x @ w - y
            g = x.T @ e / len(y)
            if w_prev is not None:
                old = x @ w_prev
                new = x @ w
                g = g + lam * (x.T @ (new - old) / len(y))
            w -= lr * np.clip(g, -50, 50)
        w_prev = w.copy()
    retain = float(np.mean((tasks[0][0] @ w - tasks[0][1]) ** 2))
    w_sgd = np.zeros(4)
    for x, y in tasks:
        for _ in range(300):
            w_sgd -= 0.05 * np.clip(x.T @ (x @ w_sgd - y) / len(y), -50, 50)
    retain_sgd = float(np.mean((tasks[0][0] @ w_sgd - tasks[0][1]) ** 2))
    return {
        "synthetic_lwf_task1_mse": retain,
        "synthetic_lwf_sgd_task1_mse": retain_sgd,
        "synthetic_lwf_retention_gain": retain_sgd - retain,
    }
