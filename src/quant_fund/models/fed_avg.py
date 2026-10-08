"""Federated averaging (FedAvg) across synthetic venues (Exec-Summary (SYNTHETIC)
distributed item). K venues each hold private order-flow shards; a global
signal model is trained by averaging local SGD updates — data never leaves
the venue. Also ships a FedProx-style proximal term and per-client drift
metrics.

Synthetic bench: heterogeneous venue shards (different microstructure
biases); FedAvg global model nearly matches centralized training and beats
any single venue's local model.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray

_N_CLIENTS = 4


def synth_shards(T: int, rng: np.random.Generator) -> list[tuple[FloatArray, FloatArray]]:
    """Each venue: same true beta but different feature distribution/scale."""
    beta = np.array([0.8, -0.5, 0.3, 0.6])
    shards = []
    for c in range(_N_CLIENTS):
        scale = 0.5 + c * 0.4
        x = rng.standard_normal((T, 4)) * scale + c * 0.2
        y = x @ beta + 0.4 * rng.standard_normal(T)
        shards.append((x, y))
    return shards


def local_train(
    x: FloatArray, y: FloatArray, w0: FloatArray, lr: float, epochs: int, mu: float = 0.0
) -> FloatArray:
    w = w0.copy()
    for _ in range(epochs):
        e = x @ w - y
        g = x.T @ e / len(y) + mu * (w - w0)
        w -= lr * g
    return w


def eval_all(w: FloatArray, shards: list[tuple[FloatArray, FloatArray]]) -> float:
    return float(np.mean([np.mean((x @ w - y) ** 2) for x, y in shards]))


def bench_fed_avg(seed: int = 17) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shards = synth_shards(60, rng)
    glob = np.zeros(4)
    rounds = 40
    drift_hist = []
    for _ in range(rounds):
        locals_ = [local_train(x, y, glob, 0.05, 3) for x, y in shards]
        drift_hist.append(float(np.mean([np.linalg.norm(loc - glob) for loc in locals_])))
        glob = np.mean(locals_, axis=0)
    fed_mse = eval_all(glob, shards)
    # centralized upper bound
    xc = np.vstack([s[0] for s in shards])
    yc = np.concatenate([s[1] for s in shards])
    wc = np.asarray(np.linalg.solve(xc.T @ xc + 1e-6 * np.eye(4), xc.T @ yc))
    cent_mse = eval_all(wc, shards)
    local_mses = [eval_all(local_train(x, y, np.zeros(4), 0.05, 40), shards) for x, y in shards]
    return {
        "synthetic_fedavg_mse": fed_mse,
        "synthetic_fedavg_central_mse": cent_mse,
        "synthetic_fedavg_gap_to_central": fed_mse - cent_mse,
        "synthetic_fedavg_best_local_mse": float(np.min(local_mses)),
        "synthetic_fedavg_margin_vs_local": float(np.min(local_mses) - fed_mse),
        "synthetic_fedavg_final_drift": drift_hist[-1],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_fed_avg(), indent=1))
