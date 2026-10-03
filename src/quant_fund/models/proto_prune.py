"""Prototypicality pruning (Sorscher et al. 2022) — keep examples
closest to class centroids (high prototypicality) vs random; on
label-noise data, centroid-distance is a mislabel detector.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dc_synth import dc_data, fit_eval


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bench_proto_prune(seed: int = 1833, keep_frac: float = 0.7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X, y, Xt, yt = dc_data(seed)
    n, d = X.shape
    rng2 = np.random.default_rng(seed)
    _ = rng2.standard_normal((n, d))
    w_true = rng2.standard_normal(d)
    w_true /= np.linalg.norm(w_true)
    p_true = 1.0 / (1.0 + np.exp(-(X @ w_true) * 3))
    y_true = (rng2.random(n) < p_true).astype(np.int64)
    _ = rng2.random(n)
    mislabeled = (y != y_true).astype(np.int64)
    proto = np.zeros(n)
    for c in (0, 1):
        c_x = X[y == c]
        cen = c_x.mean(0)
        dist = np.linalg.norm(X - cen, axis=1)
        proto[y == c] = -dist[y == c]
    auc = _auc(proto, 1 - mislabeled)  # prototypical → not mislabeled
    k = int(keep_frac * n)
    keep = np.argsort(-proto)[:k]
    acc_p = fit_eval(X[keep], y[keep], Xt, yt)
    keep2 = rng.choice(n, k, replace=False)
    acc_r = fit_eval(X[keep2], y[keep2], Xt, yt)
    acc_full = fit_eval(X, y, Xt, yt)
    return {
        "synthetic_proto_clean_auc": auc,
        "synthetic_proto_pruned_acc": acc_p,
        "synthetic_proto_random_acc": acc_r,
        "synthetic_proto_full_acc": acc_full,
        "synthetic_proto_gain": acc_p - acc_r,
        "torch_available": 0.0,
    }
