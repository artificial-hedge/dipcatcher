"""k-center coreset selection (Sener & Savarese 2018) — cover the (SYNTHETIC)
unlabeled pool greedily in feature space: pick the point farthest from
the labeled set (maximin). Accuracy vs random baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._al_synth import al_loop, random_baseline


def _select(uP, u_idx, lP, ly, w):
    # distance from each unlabeled to nearest labeled
    d = np.sqrt(((uP[:, None] - lP[None]) ** 2).sum(-1)).min(1)
    return np.argsort(-d)


def bench_coreset_kcenter(seed: int = 2619, trials: int = 5) -> dict[str, float]:
    accs = [al_loop(_select, seed=seed + 2 * t) for t in range(trials)]
    bases = [random_baseline(seed=seed + 2 * t) for t in range(trials)]
    return {
        "synthetic_kc_acc": float(np.mean(accs)),
        "synthetic_random_acc": float(np.mean(bases)),
        "synthetic_kc_gain": float(np.mean(accs) - np.mean(bases)),
        "synthetic_torch_available": 0.0,
    }
