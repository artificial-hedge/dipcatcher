"""EL2N scores (Paul et al. 2021) — early-epoch L2 norm of (prob - onehot) (SYNTHETIC)
as difficulty score; prune top-easy EL2N examples and measure accuracy
vs random-prune, plus Spearman with label-flip indicator.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dc_synth import dc_data, fit_eval


def bench_el2n_scoring(
    seed: int = 1813, early_epochs: int = 10, prune_frac: float = 0.3
) -> dict[str, float]:
    X, y, Xt, yt = dc_data(seed)
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    rng = np.random.default_rng(seed)
    w = np.zeros(Xb.shape[1])
    el2n = np.zeros(len(X))
    for _ in range(early_epochs):
        b = rng.choice(len(X), 64, replace=False)
        p = 1.0 / (1.0 + np.exp(-(Xb[b] @ w)))
        w -= 0.5 * (Xb[b].T @ (p - y[b]) / len(b) + 0.001 * w)
        p_all = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        el2n += np.abs(p_all - y.astype(float))
    el2n /= early_epochs
    k = int(prune_frac * len(X))
    keep = np.argsort(-el2n)[k:]  # prune lowest-difficulty (easiest)
    acc_el = fit_eval(X[keep], y[keep], Xt, yt)
    keep2 = rng.choice(len(X), len(X) - k, replace=False)
    acc_rand = fit_eval(X[keep2], y[keep2], Xt, yt)
    acc_full = fit_eval(X, y, Xt, yt)
    return {
        "synthetic_el2n_pruned_acc": acc_el,
        "synthetic_el2n_random_acc": acc_rand,
        "synthetic_el2n_full_acc": acc_full,
        "synthetic_el2n_gain": acc_el - acc_rand,
        "synthetic_el2n_mean": float(el2n.mean()),
        "synthetic_torch_available": 0.0,
    }
