"""Query-by-committee (Seung-Opper-Sompolinsky 1992) — bootstrap
committee of logreg models; query points of maximum vote entropy
(disagreement). Accuracy vs random baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._al_synth import al_loop, fit_logreg, random_baseline

K = 7


def _select(uP, u_idx, lP, ly, w):
    rng = np.random.default_rng(int(u_idx[0]) if len(u_idx) else 0)
    votes = np.zeros(len(uP))
    for _ in range(K):
        bs = rng.integers(0, len(lP), len(lP))
        wk = fit_logreg(lP[bs], ly[bs], iters=60)
        # committee label from bootstrapped model on unlabeled
        votes += (uP @ wk > 0).astype(np.float64)
    v = votes / K
    ent = -(v * np.log(np.clip(v, 1e-12, 1)) + (1 - v) * np.log(np.clip(1 - v, 1e-12, 1)))
    return np.argsort(-ent)


def bench_qbc_committee(seed: int = 2613, trials: int = 4) -> dict[str, float]:
    accs = [al_loop(_select, seed=seed + 2 * t) for t in range(trials)]
    bases = [random_baseline(seed=seed + 2 * t) for t in range(trials)]
    return {
        "synthetic_qbc_acc": float(np.mean(accs)),
        "synthetic_random_acc": float(np.mean(bases)),
        "synthetic_qbc_gain": float(np.mean(accs) - np.mean(bases)),
        "synthetic_torch_available": 0.0,
    }
