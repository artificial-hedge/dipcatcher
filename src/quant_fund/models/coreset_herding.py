"""Coreset selection — greedy herding (Welling 2009) (SYNTHETIC).

Greedy selection of k samples that best cover the class-conditional
feature mean; model trained on the coreset beats a random k-subset —
the data-efficiency claim.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._data_synth import synth_dataset


def _herd_select(x_pool: np.ndarray, k: int) -> list[int]:
    """Greedy herding (Welling 2009): each pick maximizes the inner product
    with the residual target-mean minus running-selected-mean, so the
    coreset's mean tracks the pool mean."""
    mu = x_pool.mean(0)
    sel: list[int] = []
    mean = np.zeros(x_pool.shape[1])
    for _j in range(k):
        residual = mu - mean
        scores = x_pool @ residual
        scores[sel] = -np.inf
        i = int(np.argmax(scores))
        sel.append(i)
        mean = x_pool[sel].mean(0)
    return sel


def bench_coreset_herding(
    seed: int = 307,
    n: int = 600,
    k: int = 30,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x, y, _h = synth_dataset(n, rng)
    cut = n // 2
    x_tr, y_tr = x[:cut], y[:cut]
    x_te, y_te = x[cut:], y[cut:]
    sel = _herd_select(x_tr, k)
    acc_h = LogisticRegression(max_iter=300).fit(x_tr[sel], y_tr[sel]).score(x_te, y_te)
    accs = []
    for s in range(5):
        idx = np.random.default_rng(seed + s).choice(cut, k, replace=False)
        accs.append(LogisticRegression(max_iter=300).fit(x_tr[idx], y_tr[idx]).score(x_te, y_te))
    acc_r = float(np.mean(accs))
    acc_f = LogisticRegression(max_iter=300).fit(x_tr, y_tr).score(x_te, y_te)
    return {
        "synthetic_herd_acc": float(acc_h),
        "synthetic_herd_random_acc": acc_r,
        "synthetic_herd_full_acc": float(acc_f),
        "synthetic_herd_gain": float(acc_h - acc_r),
    }
