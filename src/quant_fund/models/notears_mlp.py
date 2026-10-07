"""NOTEARS-MLP (Zheng et al. 2020) — nonlinear SEM recovery: each (SYNTHETIC)
variable's conditional is a small MLP; acyclicity on the weight
product of first layers. SHD vs corr baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cs_synth import corr_baseline, sem_data, shd

FloatArray = NDArray[np.float64]


def bench_notears_mlp(seed: int = 2335, edges: int = 7, steps: int = 350) -> dict[str, float]:
    X, B = sem_data(seed, edges=edges)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    H = 8
    # per-output MLP: X → H → scalar; adjacency proxy = |W1| column norms
    W1 = rng.standard_normal((d, d, H)) * 0.1  # [input, output, hidden]
    W2 = rng.standard_normal((d, H)) * 0.1  # [output, hidden]
    lam, rho = 0.0, 1.0
    lr = 0.01
    for it in range(steps):
        # forward: y_j = tanh(X @ W1[:,j,:]) @ W2[j]
        A = np.einsum("ni,ijh->njh", X, W1)
        Ht = np.tanh(A)  # n,d,H
        pred = np.einsum("njh,jh->nj", Ht, W2)
        R = pred - X
        # grads
        dH = np.einsum("nj,jh->njh", R, W2) * (1 - Ht**2)
        gW1 = np.einsum("ni,njh->ijh", X, dH) / n
        gW2 = np.einsum("njh,nj->jh", Ht, R) / n
        # acyclicity on G_ij = ||W1[i,j,:]|| (first-layer weight magnitude)
        G = np.sqrt((W1**2).sum(-1) + 1e-9)
        T = np.eye(d) + G
        Gp = G.copy()
        for k in range(2, 20):
            Gp = Gp @ G / k
            T = T + Gp
        h = float(np.trace(T)) - d
        dG = T.T
        gadj = np.einsum("ij,ijh->ijh", dG, W1) / np.sqrt((W1**2).sum(-1, keepdims=True) + 1e-9)
        grad1 = gW1 + 0.01 * np.sign(W1) + (lam + rho * h) * gadj
        W1 = W1 - lr * grad1
        W2 = W2 - lr * (gW2 + 0.01 * np.sign(W2))
        if it % 100 == 99:
            lam += rho * h
            rho = min(rho * 1.5, 1e4)
        W1 = np.nan_to_num(W1, nan=0.0)
    Bhat = np.sqrt((W1**2).sum(-1))
    Bhat[Bhat < 0.3] = 0
    shd_m = shd(B, Bhat)
    Bb = corr_baseline(X, edges)
    shd_cb = shd(B, Bb)
    return {
        "synthetic_ntmlp_shd": float(shd_m),
        "synthetic_corr_shd": float(shd_cb),
        "synthetic_ntmlp_gain": float(shd_cb - shd_m),
        "synthetic_torch_available": 0.0,
    }
