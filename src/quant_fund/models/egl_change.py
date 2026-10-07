"""Expected-gradient-length (Settles et al. 2007) — query the point
whose label would change the model most: ||φ_i||·p(1−p) (expected
gradient magnitude over both label outcomes). Accuracy vs random.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._al_synth import al_loop, probs, random_baseline


def _select(uP, u_idx, lP, ly, w):
    p = probs(uP, w)
    # E_y[||g_i(y)||] = p·||φ(1−p)|| + (1−p)·||φ·p|| = ||φ||·p(1−p)·2
    egl = np.linalg.norm(uP, axis=1) * p * (1 - p)
    return np.argsort(-egl)


def bench_egl_change(seed: int = 2631, trials: int = 5) -> dict[str, float]:
    accs = [al_loop(_select, seed=seed + 2 * t) for t in range(trials)]
    bases = [random_baseline(seed=seed + 2 * t) for t in range(trials)]
    return {
        "synthetic_egl_acc": float(np.mean(accs)),
        "synthetic_random_acc": float(np.mean(bases)),
        "synthetic_egl_gain": float(np.mean(accs) - np.mean(bases)),
        "synthetic_torch_available": 0.0,
    }
