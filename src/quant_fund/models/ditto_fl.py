"""Ditto (Li et al. 2021) — per-client personalized model trained with a (SYNTHETIC)
proximal pull toward the global FedAvg model: local accuracy vs shared
and purely-local baselines on hetero shards.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.fed_avg import synth_shards


def bench_ditto_fl(
    seed: int = 697,
    T: int = 120,
    rounds: int = 15,
    epochs: int = 4,
    lr: float = 0.05,
    mu: float = 0.5,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shards = synth_shards(T, rng)
    rng2 = np.random.default_rng(seed + 3)
    shards_te = synth_shards(40, rng2)
    # global FedAvg model
    w_g = np.zeros(4)
    for _r in range(rounds):
        ws = []
        for x, y in shards:
            wc = w_g.copy()
            for _e in range(epochs):
                wc -= lr * (x.T @ (x @ wc - y) / len(y))
            ws.append(wc)
        w_g = np.mean(ws, 0)
    # Ditto personalized: local model with prox to w_g
    vs = []
    for x, y in shards:
        v = w_g.copy()
        for _e in range(epochs * rounds):
            v -= lr * (x.T @ (x @ v - y) / len(y) + mu * (v - w_g))
        vs.append(v)

    def mse(w, shards_ev):
        return float(np.mean([(x @ w - y).var() for x, y in shards_ev]))

    err_d = float(np.mean([mse(v, shards_te) for v in vs]))
    err_g = mse(w_g, shards_te)
    # purely local (no federation)
    vs_l = []
    for x, y in shards:
        v = np.zeros(4)
        for _e in range(epochs * rounds):
            v -= lr * (x.T @ (x @ v - y) / len(y))
        vs_l.append(v)
    err_l = float(np.mean([mse(v, shards_te) for v in vs_l]))
    return {
        "synthetic_ditto_mse": err_d,
        "synthetic_ditto_global_mse": err_g,
        "synthetic_ditto_local_mse": err_l,
        "synthetic_ditto_gain": err_l - err_d,
    }
