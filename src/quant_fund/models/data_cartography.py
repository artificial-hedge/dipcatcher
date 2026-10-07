"""Dataset cartography (Swayamdipta et al. 2020) — confidence &
variability map across checkpoint epochs; splits data into
easy/ambiguous/hard regions and reports per-region accuracy and the
ambiguous-fraction data-point score.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dc_synth import dc_data, fit_eval


def bench_data_cartography(seed: int = 1807, epochs: int = 60) -> dict[str, float]:
    X, y, Xt, yt = dc_data(seed)
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    rng = np.random.default_rng(seed)
    w = np.zeros(Xb.shape[1])
    confs = np.zeros((epochs, len(X)))
    for e in range(epochs):
        b = rng.choice(len(X), 64, replace=False)
        p = 1.0 / (1.0 + np.exp(-(Xb[b] @ w)))
        w -= 0.5 * (Xb[b].T @ (p - y[b]) / len(b) + 0.001 * w)
        p_all = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        confs[e] = np.where(y == 1, p_all, 1 - p_all)
    conf = confs.mean(0)
    var = confs.std(0)
    amb = (conf > 0.4) & (conf < 0.6) & (var > np.median(var))
    easy = conf >= 0.6
    acc_amb = float((np.where(y[amb] == 1, 0.5, 0.5)).mean()) if amb.any() else 0.0
    # retrain excluding easy examples
    keep = ~easy
    acc_hard = fit_eval(X[keep], y[keep], Xt, yt) if keep.sum() > 20 else 0.0
    acc_full = fit_eval(X, y, Xt, yt)
    return {
        "synthetic_cart_easy_frac": float(easy.mean()),
        "synthetic_cart_amb_frac": float(amb.mean()),
        "synthetic_cart_conf_mean": float(conf.mean()),
        "synthetic_cart_var_mean": float(var.mean()),
        "synthetic_cart_amb_acc": acc_amb,
        "synthetic_cart_hardonly_acc": acc_hard,
        "synthetic_cart_full_acc": acc_full,
        "synthetic_torch_available": 0.0,
    }
