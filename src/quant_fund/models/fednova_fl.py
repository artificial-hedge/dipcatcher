"""FedNova (Wang et al. 2020) — normalized averaging: scale each client's (SYNTHETIC)
update by the mean local-step count to cancel objective inconsistency —
vs raw FedAvg under heterogeneous epoch budgets.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.fed_avg import synth_shards


def bench_fednova_fl(
    seed: int = 693,
    T: int = 120,
    rounds: int = 15,
    lr: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shards = synth_shards(T, rng)
    beta = np.array([0.8, -0.5, 0.3, 0.6])
    rng_e = np.random.default_rng(seed + 1)
    epochs_vec = rng_e.integers(1, 6, len(shards))  # hetero budgets
    # FedAvg (biased): average post-local w
    w_fa = np.zeros(4)
    for _r in range(rounds):
        ws = []
        for (x, y), e in zip(shards, epochs_vec, strict=True):
            wc = w_fa.copy()
            for _i in range(e):
                wc -= lr * (x.T @ (x @ wc - y) / len(y))
            ws.append(wc)
        w_fa = np.mean(ws, 0)
    # FedNova: average normalized deltas (Δ/E_i × Ē)
    w_fn = np.zeros(4)
    e_bar = epochs_vec.mean()
    for _r in range(rounds):
        deltas = []
        for (x, y), e in zip(shards, epochs_vec, strict=True):
            wc = w_fn.copy()
            for _i in range(e):
                wc -= lr * (x.T @ (x @ wc - y) / len(y))
            deltas.append((wc - w_fn) / e * e_bar)
        w_fn = w_fn + np.mean(deltas, 0)
    return {
        "synthetic_fednova_err": float(np.linalg.norm(w_fn - beta)),
        "synthetic_fednova_fedavg_err": float(np.linalg.norm(w_fa - beta)),
        "synthetic_fednova_gain": float(np.linalg.norm(w_fa - beta) - np.linalg.norm(w_fn - beta)),
    }
