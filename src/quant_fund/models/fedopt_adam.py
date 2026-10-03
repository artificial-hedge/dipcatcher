"""FedOpt-Adam (Reddi et al. 2021) — treat the averaged client delta as a
pseudo-gradient and apply server-side Adam — vs vanilla FedAvg aggregation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.fed_avg import synth_shards


def bench_fedopt_adam(
    seed: int = 703,
    T: int = 120,
    rounds: int = 15,
    epochs: int = 4,
    lr_local: float = 0.05,
    lr_server: float = 0.1,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shards = synth_shards(T, rng)
    beta = np.array([0.8, -0.5, 0.3, 0.6])
    # FedOpt: server Adam on -Δ
    w = np.zeros(4)
    m = np.zeros(4)
    v = np.zeros(4)
    for t in range(1, rounds + 1):
        deltas = []
        for x, y in shards:
            wc = w.copy()
            for _e in range(epochs):
                wc -= lr_local * (x.T @ (x @ wc - y) / len(y))
            deltas.append(wc - w)
        d = np.mean(deltas, 0)
        m = 0.9 * m + 0.1 * d
        v = 0.99 * v + 0.01 * d * d
        w = w + lr_server * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.99**t)) + 1e-4)
    err_fo = float(np.linalg.norm(w - beta))
    # plain FedAvg (server lr 1.0)
    w_fa = np.zeros(4)
    for _r in range(rounds):
        ws = []
        for x, y in shards:
            wc = w_fa.copy()
            for _e in range(epochs):
                wc -= lr_local * (x.T @ (x @ wc - y) / len(y))
            ws.append(wc)
        w_fa = np.mean(ws, 0)
    err_fa = float(np.linalg.norm(w_fa - beta))
    return {
        "synthetic_fedopt_err": err_fo,
        "synthetic_fedopt_fedavg_err": err_fa,
        "synthetic_fedopt_gain": err_fa - err_fo,
    }
