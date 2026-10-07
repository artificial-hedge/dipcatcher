"""Sliding-window KV cache (ring buffer, Mistral-style) (SYNTHETIC).

The KV cache holds only the last W tokens — memory W·d vs T·d. On a
local-dependence fixture (label depends on the last L tokens, L < W)
the windowed cache loses nothing vs full history; on long-range recall
it honestly fails.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bench_sliding_window_cache(
    seed: int = 163,
    n_seq: int = 300,
    t: int = 64,
    d: int = 8,
    window: int = 8,
    n_classes: int = 2,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_seq, t, d))
    # label = sign of mean of last-4 tokens (local task)
    y_local = (x[:, -4:].sum((-1, -2)) > 0).astype(np.int64)
    # label = sign of first-token feature (long-range task)
    y_long = (x[:, 0, 0] > 0).astype(np.int64)

    def window_feats(xb, w):
        return xb[:, -w:].reshape(xb.shape[0], -1)

    def full_feats(xb):
        return xb.reshape(xb.shape[0], -1)

    from numpy.linalg import lstsq

    def fit_acc(feats, yv):
        wv, *_ = lstsq(feats, np.where(yv == 1, 1.0, -1.0), rcond=None)
        return float(((feats @ wv > 0).astype(np.int64) == yv).mean())

    acc_w_local = fit_acc(window_feats(x, window), y_local)
    acc_f_local = fit_acc(full_feats(x), y_local)
    acc_w_long = fit_acc(window_feats(x, window), y_long)
    acc_f_long = fit_acc(full_feats(x), y_long)
    mem_frac = float(window) / t
    return {
        "synthetic_swc_local_window_acc": acc_w_local,
        "synthetic_swc_local_full_acc": acc_f_local,
        "synthetic_swc_local_gap": acc_f_local - acc_w_local,
        "synthetic_swc_long_window_acc": acc_w_long,
        "synthetic_swc_long_full_acc": acc_f_long,
        "synthetic_swc_long_gap": acc_f_long - acc_w_long,
        "synthetic_swc_mem_frac": mem_frac,
    }
