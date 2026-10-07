"""Entropy uncertainty sampling (Shannon 1948; Settles 2009) — query
the unlabeled points whose predicted class posterior is closest to
uniform. AL loop accuracy vs random-selection baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._al_synth import al_loop, probs, random_baseline


def _select(uP, u_idx, lP, ly, w):
    p = probs(uP, w)
    ent = -(p * np.log(np.clip(p, 1e-12, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-12, 1)))
    return np.argsort(-ent)[: len(uP)]


def bench_entropy_query(seed: int = 2601, trials: int = 5) -> dict[str, float]:
    accs = [al_loop(_select, seed=seed + 2 * t) for t in range(trials)]
    bases = [random_baseline(seed=seed + 2 * t) for t in range(trials)]
    return {
        "synthetic_eq_acc": float(np.mean(accs)),
        "synthetic_random_acc": float(np.mean(bases)),
        "synthetic_eq_gain": float(np.mean(accs) - np.mean(bases)),
        "synthetic_torch_available": 0.0,
    }
