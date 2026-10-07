"""A-GEM (Chaudhry et al. 2019) — average GEM: project each update's (SYNTHETIC)
gradient so it doesn't increase loss on a small episodic memory (dot-product
projection against memory gradient).
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.continual_learning import regime_panel


def _agem_projected(g: np.ndarray, gref: np.ndarray | None) -> np.ndarray:
    """Return the gradient actually applied by A-GEM: clip, then project.

    Clipping must precede the projection check — clipping after the
    projection can re-introduce a negative g·gref and silently void the
    non-interference guarantee.
    """
    g = np.clip(g, -50, 50)
    if gref is not None and float(g @ gref) < 0:
        g = g - (g @ gref) / max(float(gref @ gref), 1e-12) * gref
    return np.asarray(g)


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
                gref: np.ndarray | None = Xm.T @ (Xm @ w - ym) / len(Xm)
            else:
                gref = None
            w -= lr * _agem_projected(g, gref)
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
