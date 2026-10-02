"""DAG-GNN (Yu et al. 2019) — graph neural net VAE for structure:
encoder infers adjacency logits, decoder f(X) = (I-A^T)^{-1} f0(X)X;
acyclicity via tr((I+A)^{d-1}). SHD vs corr baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cs_synth import corr_baseline, sem_data, shd

FloatArray = NDArray[np.float64]


def bench_dag_gnn(seed: int = 2341, edges: int = 7, steps: int = 350) -> dict[str, float]:
    X, B = sem_data(seed, edges=edges)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    A = np.zeros((d, d))  # adjacency logits
    W1 = rng.standard_normal((d, d)) * 0.1  # decoder weight
    lam, rho = 0.0, 1.0
    lr = 0.02
    for it in range(steps):
        P = 1 / (1 + np.exp(-np.clip(A, -30, 30)))
        # decoder: X_hat ≈ X @ (I + P^T) @ W1 (first-order inverse approx)
        G = (np.eye(d) + P.T) @ W1
        pred = X @ G
        R = pred - X
        gW1 = (np.eye(d) + P.T).T @ (X.T @ R) / n
        gG = X.T @ R / n
        gP = W1 @ gG.T * P * (1 - P)  # through sigmoid
        # acyclicity h = tr((I+P)^{d-1}) - d
        M = np.linalg.matrix_power(np.eye(d) + P, d - 1)
        h = float(np.trace(M)) - d
        gh = (d - 1) * np.linalg.matrix_power(np.eye(d) + P, d - 2).T * P * (1 - P)
        gradA = gP.T + 0.01 * np.sign(A) + (lam + rho * h) * gh
        A = A - lr * gradA
        W1 = W1 - lr * (gW1 + 0.01 * np.sign(W1))
        if it % 100 == 99:
            lam += rho * h
            rho = min(rho * 1.5, 1e4)
        A = np.nan_to_num(A, nan=0.0)
    Bhat = (1 / (1 + np.exp(-np.clip(A, -30, 30))) > 0.5).astype(float)
    shd_gnn = shd(B, Bhat)
    Bb = corr_baseline(X, edges)
    shd_cb = shd(B, Bb)
    return {
        "synthetic_daggnn_shd": float(shd_gnn),
        "synthetic_corr_shd": float(shd_cb),
        "synthetic_daggnn_gain": float(shd_cb - shd_gnn),
        "torch_available": 0.0,
    }
