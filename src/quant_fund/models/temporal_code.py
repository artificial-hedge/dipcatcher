"""Temporal coding comparison — rank-order (latency) vs rate (Poisson
count) coding on the same classifier: information per spike (acc /
mean spikes) favors latency coding.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sk_synth import latency_encode, poisson_encode, sk_data


def _fit(Xb: np.ndarray, y: np.ndarray, iters: int = 300) -> np.ndarray:
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        w -= 0.3 * (Xb.T @ (p - y) / len(y) + 0.001 * w)
    return w


def bench_temporal_code(seed: int = 1933, T: int = 30) -> dict[str, float]:
    X, y = sk_data(seed)
    Xt, yt = sk_data(seed + 1, n=200)
    sp_r = poisson_encode(X, seed=seed)[:, :, :T].sum(-1)
    sp_rt = poisson_encode(Xt, seed=seed + 1)[:, :, :T].sum(-1)
    sp_l = latency_encode(X, T)
    sp_lt = latency_encode(Xt, T)

    # latency feature: T - first spike time (earlier = stronger)
    def lat_feat(sp):
        idx = np.argmax(sp > 0, axis=-1)
        has = (sp > 0).any(-1)
        return np.where(has, T - idx, 0.0)

    fl = lat_feat(sp_l)
    flt = lat_feat(sp_lt)
    wr = _fit(np.concatenate([sp_r, np.ones((len(sp_r), 1))], 1), y)
    acc_r = float(
        ((np.concatenate([sp_rt, np.ones((len(sp_rt), 1))], 1) @ wr > 0).astype(int) == yt).mean()
    )
    wl = _fit(np.concatenate([fl, np.ones((len(fl), 1))], 1), y)
    acc_l = float(
        ((np.concatenate([flt, np.ones((len(flt), 1))], 1) @ wl > 0).astype(int) == yt).mean()
    )
    spikes_r = float(poisson_encode(X, seed=seed)[:, :, :T].mean() * T)
    spikes_l = 1.0  # exactly one spike per input
    return {
        "synthetic_rate_acc": acc_r,
        "synthetic_latency_acc": acc_l,
        "synthetic_rate_spikes_per_input": spikes_r,
        "synthetic_latency_spikes_per_input": spikes_l,
        "synthetic_latency_efficiency": acc_l / spikes_l - acc_r / spikes_r,
        "torch_available": 0.0,
    }
