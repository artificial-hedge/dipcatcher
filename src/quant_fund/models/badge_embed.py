"""BADGE sampling (Ash et al. 2020) — gradient-embedding k-means++: (SYNTHETIC)
hypothetical-label loss gradients g_i = φ_i·(p_i − ŷ_i) seed diverse
uncertain queries. Accuracy vs random baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._al_synth import al_loop, probs, random_baseline


def _select(uP, u_idx, lP, ly, w):
    rng = np.random.default_rng(int(u_idx[0]) if len(u_idx) else 0)
    p = probs(uP, w)
    yhat = (p > 0.5).astype(np.float64)
    G = uP * (p - yhat)[:, None]  # gradient embeddings
    # k-means++ over embeddings
    chosen = [int(rng.integers(len(G)))]
    for _ in range(min(8, len(G) - 1)):
        d2 = np.min(((G[:, None] - G[None, chosen]) ** 2).sum(-1), 1)
        chosen.append(int(np.argmax(d2)))
    order = list(chosen)
    rest = [i for i in range(len(G)) if i not in order]
    return np.array(order + rest)


def bench_badge_embed(seed: int = 2625, trials: int = 5) -> dict[str, float]:
    accs = [al_loop(_select, seed=seed + 2 * t) for t in range(trials)]
    bases = [random_baseline(seed=seed + 2 * t) for t in range(trials)]
    return {
        "synthetic_badge_acc": float(np.mean(accs)),
        "synthetic_random_acc": float(np.mean(bases)),
        "synthetic_badge_gain": float(np.mean(accs) - np.mean(bases)),
        "synthetic_torch_available": 0.0,
    }
