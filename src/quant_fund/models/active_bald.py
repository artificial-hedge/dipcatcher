"""BALD active learning (Houlsby et al. 2011) — query-by-committee
disagreement (5 bagged logistic voters): select top-k highest
vote-entropy samples each round vs random selection, same budget.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dc_synth import dc_data, fit_eval


def _fit_w(Xb: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.zeros(Xb.shape[1])
    idx = np.arange(len(y))
    for _ in range(150):
        b = rng.choice(idx, min(64, len(y)), replace=False)
        p = 1.0 / (1.0 + np.exp(-(Xb[b] @ w)))
        w -= 0.5 * (Xb[b].T @ (p - y[b]) / len(b) + 0.001 * w)
    return w


def bench_active_bald(seed: int = 1801, budget: int = 60, rounds: int = 3) -> dict[str, float]:
    X, y, Xt, yt = dc_data(seed)
    rng = np.random.default_rng(seed)
    per = budget // rounds
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    # BALD-style: committee of 5 voters on unlabeled pool
    sel = np.zeros(len(X), bool)
    for _ in range(rounds):
        voters = np.stack([_fit_w(Xb[sel], y[sel], rng) for _ in range(5)]) if sel.any() else None
        if voters is None:
            pick = rng.choice(len(X), per, replace=False)
        else:
            probs = 1.0 / (1.0 + np.exp(-(voters @ Xb.T)))  # (5, n)
            mean_p = probs.mean(0)
            h_mean = -mean_p * np.log(mean_p + 1e-9) - (1 - mean_p) * np.log(1 - mean_p + 1e-9)
            h_each = (-probs * np.log(probs + 1e-9) - (1 - probs) * np.log(1 - probs + 1e-9)).mean(
                0
            )
            bald = h_mean - h_each
            bald[sel] = -1
            pick = np.argsort(-bald)[:per]
        sel[pick] = True
    acc_bald = fit_eval(X[sel], y[sel], Xt, yt)
    sel2 = np.zeros(len(X), bool)
    for _ in range(rounds):
        pool = np.where(~sel2)[0]
        sel2[rng.choice(pool, per, replace=False)] = True
    acc_rand = fit_eval(X[sel2], y[sel2], Xt, yt)
    acc_full = fit_eval(X, y, Xt, yt)
    return {
        "synthetic_bald_acc": acc_bald,
        "synthetic_random_acc": acc_rand,
        "synthetic_full_acc": acc_full,
        "synthetic_bald_gain": acc_bald - acc_rand,
        "synthetic_torch_available": 0.0,
    }
