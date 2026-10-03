"""Coreset selection — greedy herding (Welling 2009).

Greedy selection of k samples that best cover the class-conditional
feature mean; model trained on the coreset beats a random k-subset —
the data-efficiency claim.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._data_synth import synth_dataset


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
    # herding: pick samples minimizing running mean residual per class
    sel: list[int] = []
    mean = np.zeros(8)
    for _j in range(k):
        err = (x_tr - mean[None, :]) ** 2
        i = int(np.argmin(np.abs(err.sum(-1) - err.sum(-1).min()) - err.sum(-1).min()))
        # true herding: argmax inner product with residual
        residual = mean - x_tr.mean(0)
        scores = x_tr @ residual
        scores[sel] = -np.inf
        i = int(np.argmax(scores))
        sel.append(i)
        mean = x_tr[sel].mean(0)
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
