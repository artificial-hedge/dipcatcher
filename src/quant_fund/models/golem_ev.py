"""GOLEM (Ng et al. 2020) — likelihood-based DAG recovery: Gaussian
equal-variance score (replaces least-squares BIC) + NOTEARS acyclicity.
Better under unequal variances; SHD vs corr baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cs_synth import corr_baseline, sem_data, shd

FloatArray = NDArray[np.float64]


def bench_golem_ev(seed: int = 2329, edges: int = 7, steps: int = 400) -> dict[str, float]:
    X, B = sem_data(seed, edges=edges)
    n, d = X.shape
    W = np.zeros((d, d))
    lam, rho = 0.0, 1.0
    lr = 0.005

    def score(W: FloatArray) -> tuple[float, FloatArray]:
        R = X - X @ W  # residuals
        mse = float((R * R).mean())
        # GOLEM-EV: n*d/2 log MSE + penalty; grad of MSE part
        grad = -(X.T @ R) / n / max(mse, 1e-6) * d
        return mse * d * np.log(max(mse, 1e-6)), grad

    for it in range(steps):
        _, gs = score(W)
        g = gs + 0.05 * np.sign(W)
        M = W * W
        T = np.eye(d) + M
        Mp = M.copy()
        for k in range(2, 20):
            Mp = Mp @ M / k
            T = T + Mp
        h = float(np.trace(T)) - d
        gh = 2 * W * T.T
        grad = g + lam * gh + rho * h * gh
        W = W - lr * grad
        W = np.nan_to_num(W, nan=0.0)
        if it % 100 == 99:
            lam += rho * h
            rho = min(rho * 1.5, 1e4)
    W[np.abs(W) < 0.3] = 0
    shd_g = shd(B, W)
    Bb = corr_baseline(X, edges)
    shd_cb = shd(B, Bb)
    return {
        "synthetic_golem_shd": float(shd_g),
        "synthetic_corr_shd": float(shd_cb),
        "synthetic_golem_gain": float(shd_cb - shd_g),
        "torch_available": 0.0,
    }
