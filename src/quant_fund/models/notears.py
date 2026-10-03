"""NOTEARS (Zheng et al. 2018) — continuous DAG recovery: minimize
least-squares + l1 subject to h(W) = tr(e^{W∘W}) - d = 0 via augmented
Lagrangian. SHD vs correlation-sort baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cs_synth import corr_baseline, sem_data, shd

FloatArray = NDArray[np.float64]


def _h(W: FloatArray) -> tuple[float, FloatArray]:
    M = W * W
    d = len(W)
    # tr(e^M) via series
    T = np.eye(d) + M
    Mpow = M.copy()
    for k in range(2, 20):
        Mpow = Mpow @ M / k
        T = T + Mpow
    h = float(np.trace(T)) - d
    # dh/dW = 2 W ∘ (e^M)^T
    return h, 2 * W * T.T


def bench_notears(seed: int = 2317, edges: int = 7, steps: int = 400) -> dict[str, float]:
    X, B = sem_data(seed, edges=edges)
    n, d = X.shape
    W = np.zeros((d, d))
    lam, rho = 0.0, 1.0
    lr = 0.01
    for it in range(steps):
        pred = X @ W
        g = (X.T @ (pred - X)) / n + 0.05 * np.sign(W)
        h, gh = _h(W)
        grad = g + lam * gh + rho * h * gh
        W = W - lr * grad
        if it % 100 == 99:
            h, _ = _h(W)
            lam += rho * h
            rho = min(rho * 1.5, 1e4)
    W[np.abs(W) < 0.3] = 0
    shd_nt = shd(B, W)
    Bb = corr_baseline(X, edges)
    shd_cb = shd(B, Bb)
    return {
        "synthetic_notears_shd": float(shd_nt),
        "synthetic_corr_shd": float(shd_cb),
        "synthetic_notears_gain": float(shd_cb - shd_nt),
        "torch_available": 0.0,
    }
