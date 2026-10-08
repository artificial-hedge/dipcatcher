"""Liquid state machine (Maass et al. 2002) — random recurrent spiking (SYNTHETIC)
reservoir driven by encoded inputs; linear readout on reservoir state
vs direct linear readout. Reservoir adds short-term memory.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sk_synth import sk_data


def _fit(Xb: np.ndarray, y: np.ndarray, iters: int = 300) -> np.ndarray:
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        w -= 0.3 * (Xb.T @ (p - y) / len(y) + 0.001 * w)
    return w


def bench_lsm_reservoir(seed: int = 1927, R: int = 64, T: int = 25) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X, y = sk_data(seed)
    Xt, yt = sk_data(seed + 1, n=200)
    Wr = rng.standard_normal((R, R)) * 0.1
    eigs = np.linalg.eigvals(Wr).real.max()
    Wr *= 0.5 / max(eigs, 0.5)
    Win = rng.standard_normal((8, R)) * 0.5

    def reservoir(x):
        v = np.zeros(R)
        states = []
        drive = x @ Win / (np.linalg.norm(x) * 20)
        for _ in range(T):
            v = np.clip(0.8 * v + drive + v @ Wr, -3, 3)
            s = (v > 0.5).astype(float)
            v = v * (1 - s)
            states.append(v)
        return np.asarray(states).mean(0)

    Z = np.stack([reservoir(x) for x in X])
    Zt = np.stack([reservoir(x) for x in Xt])
    w = _fit(np.concatenate([Z, np.ones((len(Z), 1))], 1), y)
    acc_l = float(
        ((np.concatenate([Zt, np.ones((len(Zt), 1))], 1) @ w > 0).astype(int) == yt).mean()
    )
    w0 = _fit(np.concatenate([X, np.ones((len(X), 1))], 1), y)
    acc_d = float(
        ((np.concatenate([Xt, np.ones((len(Xt), 1))], 1) @ w0 > 0).astype(int) == yt).mean()
    )
    return {
        "synthetic_lsm_acc": acc_l,
        "synthetic_lsm_direct_acc": acc_d,
        "synthetic_lsm_gap": acc_l - acc_d,
        "synthetic_lsm_state_std": float(Z.std()),
        "synthetic_torch_available": 0.0,
    }
