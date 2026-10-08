"""SCAFFOLD (Karimireddy et al. 2020) — client + server control variates (SYNTHETIC)
correct local-drift in FedAvg on heterogeneous shards.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.fed_avg import synth_shards


def _fedavg_rounds(shards, rounds: int, epochs: int, lr: float):
    d = shards[0][0].shape[1]
    w = np.zeros(d)
    for _r in range(rounds):
        ws = []
        for x, y in shards:
            wc = w.copy()
            for _e in range(epochs):
                wc -= lr * (x.T @ (x @ wc - y) / len(y))
            ws.append(wc)
        w = np.mean(ws, 0)
    return w


def bench_scaffold_fl(
    seed: int = 691,
    T: int = 120,
    rounds: int = 15,
    epochs: int = 4,
    lr: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shards = synth_shards(T, rng)
    beta = np.array([0.8, -0.5, 0.3, 0.6])
    # SCAFFOLD: control variate c (client), C (server)
    w = np.zeros(4)
    cs = [np.zeros(4) for _ in shards]
    C = np.zeros(4)
    for _r in range(rounds):
        deltas = []
        dcs = []
        for c_i, (x, y) in enumerate(shards):
            wc = w.copy()
            for _e in range(epochs):
                g = x.T @ (x @ wc - y) / len(y) - cs[c_i] + C
                wc -= lr * g
            # c+ = c_i - C + (w - wc)/(E·lr); deltas averaged in parallel
            c_new = cs[c_i] - C + (w - wc) / (epochs * lr)
            dcs.append(c_new - cs[c_i])
            cs[c_i] = c_new
            deltas.append(wc - w)
        w += np.mean(deltas, 0)
        C += np.mean(dcs, 0)
    err_sc = float(np.linalg.norm(w - beta))
    w_fa = _fedavg_rounds(shards, rounds, epochs, lr)
    err_fa = float(np.linalg.norm(w_fa - beta))
    return {
        "synthetic_scaffold_err": err_sc,
        "synthetic_scaffold_fedavg_err": err_fa,
        "synthetic_scaffold_gain": err_fa - err_sc,
    }
