"""STDP learning (Bi & Poo 1998) — classic selectivity demo: half the (SYNTHETIC)
inputs fire causally before the post-synaptic spike, half are
uncorrelated noise; STDP potentiates only the causal inputs.
Weight selectivity = mean W_causal - mean W_noise.
"""

from __future__ import annotations

import numpy as np


def bench_stdp_learn(seed: int = 1907, n_inputs: int = 8, T: int = 400) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # post spike train: driven by inputs 0..3 with delay 2
    post = np.zeros(T)
    pre = rng.random((n_inputs, T)) < 0.05
    # causal: inputs 0-3 spike ~2 steps before post
    pre[:4] = rng.random((4, T)) < 0.02
    drives = np.where(rng.random(T) < 0.03)[0]
    for t in drives:
        if t + 2 < T:
            post[t + 2] = 1.0
            pre[:4, t] = 1.0
    W = np.full(n_inputs, 0.5)
    A_p, A_m, tau = 0.01, 0.0105, 10.0
    pre_tr = np.zeros(n_inputs)
    post_tr = 0.0
    for tt in range(T):
        pre_tr *= np.exp(-1 / tau)
        post_tr *= np.exp(-1 / tau)
        for j in range(n_inputs):
            if pre[j, tt]:
                W[j] -= A_m * post_tr  # pre-before-post on post trace → LTD
                pre_tr[j] += 1.0
        if post[tt]:
            W += A_p * pre_tr  # pre before post → LTP
            post_tr += 1.0
        W = np.clip(W, 0.0, 1.0)
    sel = float(W[:4].mean() - W[4:].mean())
    return {
        "synthetic_stdp_selectivity": sel,
        "synthetic_stdp_w_causal": float(W[:4].mean()),
        "synthetic_stdp_w_noise": float(W[4:].mean()),
        "synthetic_torch_available": 0.0,
    }
