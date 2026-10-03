"""DAGMA (Bello et al. 2022) — log-determinant acyclicity:
h(W) = -log det(s I - W∘W) + d log s, smoother than NOTEARS's tr(e^M).
Augmented-Lagrangian least-squares recovery; SHD vs corr baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cs_synth import corr_baseline, sem_data, shd

FloatArray = NDArray[np.float64]


def _h_logdet(W: FloatArray, s: float = 1.0) -> tuple[float, FloatArray]:
    d = len(W)
    M = s * np.eye(d) - W * W
    sign, logdet = np.linalg.slogdet(M)
    h = -logdet + d * np.log(s)
    Minv = np.linalg.inv(M)
    return float(h), 2 * W * Minv.T


def bench_dagma_lin(seed: int = 2323, edges: int = 7, steps: int = 400) -> dict[str, float]:
    X, B = sem_data(seed, edges=edges)
    n, d = X.shape
    W = np.zeros((d, d))
    lam, rho = 0.0, 1.0
    lr = 0.01
    for it in range(steps):
        pred = X @ W
        g = (X.T @ (pred - X)) / n + 0.05 * np.sign(W)
        try:
            h, gh = _h_logdet(W)
        except np.linalg.LinAlgError:
            h, gh = 1.0, np.sign(W) * 0.1
        grad = g + lam * gh + rho * h * gh
        W = W - lr * grad
        W = np.nan_to_num(W, nan=0.0)
        if it % 100 == 99:
            h, _ = _h_logdet(W)
            lam += rho * h
            rho = min(rho * 1.5, 1e4)
    W[np.abs(W) < 0.3] = 0
    shd_dg = shd(B, W)
    Bb = corr_baseline(X, edges)
    shd_cb = shd(B, Bb)
    return {
        "synthetic_dagma_shd": float(shd_dg),
        "synthetic_corr_shd": float(shd_cb),
        "synthetic_dagma_gain": float(shd_cb - shd_dg),
        "torch_available": 0.0,
    }
