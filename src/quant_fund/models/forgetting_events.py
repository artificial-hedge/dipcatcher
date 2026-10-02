"""Forgetting events (Toneva et al. 2019) — count label flips of
per-example predictions across SGD epochs; scores correlate with
label noise. Bench: AUC of forgetting-count as mislabel detector +
accuracy after dropping top-forgotten.
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


def bench_forgetting_events(seed: int = 1819, epochs: int = 40) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X, y, Xt, yt = dc_data(seed)
    n, d = X.shape
    # recover true (pre-flip) labels: re-generate with same rng shape as dc_data
    rng2 = np.random.default_rng(seed)
    _ = rng2.standard_normal((n, d))
    w_true = rng2.standard_normal(d)
    w_true /= np.linalg.norm(w_true)
    p_true = 1.0 / (1.0 + np.exp(-(X @ w_true) * 3))
    y_true = (rng2.random(n) < p_true).astype(np.int64)
    _ = rng2.random(n)  # flip mask consumed
    mislabeled = (y != y_true).astype(np.int64)
    Xb = np.concatenate([X, np.ones((n, 1))], 1)
    w = np.zeros(Xb.shape[1])
    prev = np.zeros(n, bool)
    forgot = np.zeros(n)
    started = np.zeros(n, bool)
    for _ in range(epochs):
        b = rng.choice(n, 64, replace=False)
        p = 1.0 / (1.0 + np.exp(-(Xb[b] @ w)))
        w -= 0.5 * (Xb[b].T @ (p - y[b]) / len(b) + 0.001 * w)
        pred = (Xb @ w > 0) == (y == 1)
        forgot[started & ~pred & prev] += 1
        prev = pred
        started = np.ones(n, bool)
    auc = _auc(forgot, mislabeled)
    drop = forgot > np.quantile(forgot, 0.75)
    keep = ~drop
    acc_drop = fit_eval(X[keep], y[keep], Xt, yt)
    acc_full = fit_eval(X, y, Xt, yt)
    return {
        "synthetic_forget_mislabel_auc": auc,
        "synthetic_forget_pruned_acc": acc_drop,
        "synthetic_forget_full_acc": acc_full,
        "synthetic_forget_mean": float(forgot.mean()),
        "torch_available": 0.0,
    }
