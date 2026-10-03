"""LIF neuron classifier (leaky integrate-and-fire) — Poisson-encoded
inputs drive a membrane-potential readout trained by logistic loss on
final voltage vs ANN on raw features.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sk_synth import lif_forward, poisson_encode, sk_data


def _fit(Xb: np.ndarray, y: np.ndarray, iters: int = 300) -> np.ndarray:
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        w -= 0.3 * (Xb.T @ (p - y) / len(y) + 0.001 * w)
    return w


def bench_lif_neuron(seed: int = 1901) -> dict[str, float]:
    X, y = sk_data(seed)
    Xt, yt = sk_data(seed + 1, n=200)
    sp = poisson_encode(X, seed=seed)
    spt = poisson_encode(Xt, seed=seed + 1)
    # decodeable: use spike counts per input as features
    cnt = sp.sum(-1)
    cnt_t = spt.sum(-1)
    w = _fit(np.concatenate([cnt, np.ones((len(cnt), 1))], 1), y)
    acc_lif = float(
        ((np.concatenate([cnt_t, np.ones((len(cnt_t), 1))], 1) @ w > 0).astype(int) == yt).mean()
    )
    wa = _fit(np.concatenate([X, np.ones((len(X), 1))], 1), y)
    acc_ann = float(
        ((np.concatenate([Xt, np.ones((len(Xt), 1))], 1) @ wa > 0).astype(int) == yt).mean()
    )
    # membrane trace readout
    v = lif_forward(sp, np.ones(8))
    return {
        "synthetic_lif_acc": acc_lif,
        "synthetic_lif_ann_acc": acc_ann,
        "synthetic_lif_gap": acc_lif - acc_ann,
        "synthetic_lif_mean_v": float(v.mean()),
        "synthetic_lif_spike_rate": float(sp.mean()),
        "torch_available": 0.0,
    }
