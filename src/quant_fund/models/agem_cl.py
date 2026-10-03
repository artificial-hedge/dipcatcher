"""A-GEM (Chaudhry et al. 2019) — average GEM: project each update's
gradient so it doesn't increase loss on a small episodic memory (dot-product
projection against memory gradient).
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.continual_learning import regime_panel


def bench_agem_cl(seed: int = 771, T: int = 200, buf: int = 40) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kinds = ["momentum", "reversal", "volatility"]
    tasks = [regime_panel(k, T, rng) for k in kinds]
    w = np.zeros(4)
    mx: list[np.ndarray] = []
    my: list[np.ndarray] = []
    lr = 0.05
    for x, y in tasks:
        for _ in range(300):
            g = x.T @ (x @ w - y) / len(y)
            if mx:
                Xm = np.asarray(mx)
                ym = np.asarray(my)
                gref = Xm.T @ (Xm @ w - ym) / len(Xm)
                if float(g @ gref) < 0:
                    g = g - (g @ gref) / max(float(gref @ gref), 1e-12) * gref
            w -= lr * np.clip(g, -50, 50)
        idx = rng.choice(len(x), min(buf, len(x)), replace=False)
        mx = list(x[idx])
        my = list(y[idx])
    retain = float(np.mean((tasks[0][0] @ w - tasks[0][1]) ** 2))
    w_sgd = np.zeros(4)
    for x, y in tasks:
        for _ in range(300):
            w_sgd -= 0.05 * np.clip(x.T @ (x @ w_sgd - y) / len(y), -50, 50)
    retain_sgd = float(np.mean((tasks[0][0] @ w_sgd - tasks[0][1]) ** 2))
    return {
        "synthetic_agem_task1_mse": retain,
        "synthetic_agem_sgd_task1_mse": retain_sgd,
        "synthetic_agem_retention_gain": retain_sgd - retain,
    }
