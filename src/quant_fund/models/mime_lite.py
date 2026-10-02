"""MimeLite (Karimireddy et al. 2021) — server stats applied locally:
clients run local updates biased by the global full-batch gradient
estimate computed on the server side each round — vs FedAvg.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.fed_avg import synth_shards


def bench_mime_lite(
    seed: int = 707,
    T: int = 120,
    rounds: int = 15,
    epochs: int = 4,
    lr: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shards = synth_shards(T, rng)
    beta = np.array([0.8, -0.5, 0.3, 0.6])

    def full_grad(w):
        g = np.zeros(4)
        n = 0
        for x, y in shards:
            g += x.T @ (x @ w - y)
            n += len(y)
        return g / n

    # MimeLite: each client uses server full-grad as its "momentum" anchor
    w_m = np.zeros(4)
    for _r in range(rounds):
        g_srv = full_grad(w_m)
        ws = []
        for x, y in shards:
            wc = w_m.copy()
            for _e in range(epochs):
                g_loc = x.T @ (x @ wc - y) / len(y)
                wc -= lr * (g_loc + 0.5 * g_srv)  # global-stat bias
            ws.append(wc)
        w_m = np.mean(ws, 0)
    err_m = float(np.linalg.norm(w_m - beta))
    w_fa = np.zeros(4)
    for _r in range(rounds):
        ws = []
        for x, y in shards:
            wc = w_fa.copy()
            for _e in range(epochs):
                wc -= lr * (x.T @ (x @ wc - y) / len(y))
            ws.append(wc)
        w_fa = np.mean(ws, 0)
    err_fa = float(np.linalg.norm(w_fa - beta))
    return {
        "synthetic_mime_err": err_m,
        "synthetic_mime_fedavg_err": err_fa,
        "synthetic_mime_gain": err_fa - err_m,
    }
