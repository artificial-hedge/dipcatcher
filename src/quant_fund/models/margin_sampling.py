"""Margin sampling (Scheffer et al. 2001) — query points with the (SYNTHETIC)
smallest |p−0.5| decision margin; distinct from entropy by selecting
the single most ambiguous mode. Accuracy vs random baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._al_synth import al_loop, probs, random_baseline


def _select(uP, u_idx, lP, ly, w):
    p = probs(uP, w)
    return np.argsort(np.abs(p - 0.5))


def bench_margin_sampling(seed: int = 2607, trials: int = 5) -> dict[str, float]:
    accs = [al_loop(_select, seed=seed + 2 * t) for t in range(trials)]
    bases = [random_baseline(seed=seed + 2 * t) for t in range(trials)]
    return {
        "synthetic_ms_acc": float(np.mean(accs)),
        "synthetic_random_acc": float(np.mean(bases)),
        "synthetic_ms_gain": float(np.mean(accs) - np.mean(bases)),
        "synthetic_torch_available": 0.0,
    }
