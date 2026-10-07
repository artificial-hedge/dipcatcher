"""Influence functions (Koh & Liang 2017) — first-order logistic
influence approx via HVP-less diagonal-Hessian estimate; flag most
harmful points (highest negative influence on test loss) and measure
mislabel-detection AUC.
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


def bench_influence_func(seed: int = 1827) -> dict[str, float]:
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
    Xb = np.concatenate([X, np.ones((n, 1))], 1)
    w = np.zeros(Xb.shape[1])
    for _ in range(300):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        w -= 0.3 * (Xb.T @ (p - y) / n + 0.01 * w)
    p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
    # diag hessian h = p(1-p) x^2
    h = (p * (1 - p))[:, None] * Xb**2 + 0.01
    Xtb = np.concatenate([Xt, np.ones((len(Xt), 1))], 1)
    pt = 1.0 / (1.0 + np.exp(-(Xtb @ w)))
    gt = Xtb.T @ (pt - yt) / len(Xt)
    # per-train-point gradient and diag-H inverse product
    gi = Xb * (p - y)[:, None]
    infl = -(gi @ (gt / h.mean(0)))
    auc = _auc(infl, mislabeled)  # high self-influence ≈ mislabeled
    drop = infl < np.quantile(infl, 0.2)
    acc_clean = fit_eval(X[~drop], y[~drop], Xt, yt)
    acc_full = fit_eval(X, y, Xt, yt)
    return {
        "synthetic_infl_mislabel_auc": auc,
        "synthetic_infl_pruned_acc": acc_clean,
        "synthetic_infl_full_acc": acc_full,
        "synthetic_torch_available": 0.0,
    }
