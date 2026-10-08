"""Dark Experience Replay (Buzzega et al. 2020) — reservoir buffer stores (SYNTHETIC)
past (x, teacher logits); new-task loss + replay of stored points with
logit matching to the stored teacher outputs.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.continual_learning import regime_panel


def bench_der_cl(
    seed: int = 769, T: int = 200, buf: int = 40, lam: float = 1.5
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kinds = ["momentum", "reversal", "volatility"]
    tasks = [regime_panel(k, T, rng) for k in kinds]
    w = np.zeros(4)
    bx: list[np.ndarray] = []
    bl: list[np.ndarray] = []  # stored teacher logits
    lr = 0.05
    for x, y in tasks:
        for _ in range(300):
            g = x.T @ (x @ w - y) / len(y)
            if bx:
                Xb = np.asarray(bx)
                Lb = np.asarray(bl)
                g = g + lam * (Xb.T @ (Xb @ w - Lb) / len(Xb))
            w -= lr * np.clip(g, -50, 50)
        # reservoir: keep random `buf` points + current logits
        idx = rng.choice(len(x), min(buf, len(x)), replace=False)
        bx = list(x[idx])
        bl = list(x[idx] @ w)
    retain = float(np.mean((tasks[0][0] @ w - tasks[0][1]) ** 2))
    w_sgd = np.zeros(4)
    for x, y in tasks:
        for _ in range(300):
            w_sgd -= 0.05 * np.clip(x.T @ (x @ w_sgd - y) / len(y), -50, 50)
    retain_sgd = float(np.mean((tasks[0][0] @ w_sgd - tasks[0][1]) ** 2))
    return {
        "synthetic_der_task1_mse": retain,
        "synthetic_der_sgd_task1_mse": retain_sgd,
        "synthetic_der_retention_gain": retain_sgd - retain,
    }
