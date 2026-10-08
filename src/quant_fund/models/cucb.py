"""CUCB (Chen et al. 2013) — combinatorial UCB semi-bandit: pick top-m (SYNTHETIC)
arms each round by UCB on per-arm mean estimates; observe all chosen.
Expected reward vs random top-m.
"""

from __future__ import annotations

import numpy as np


def _ucb_scores(rew: np.ndarray, pull: np.ndarray, t: int) -> np.ndarray:
    """UCB1 scores; unpulled arms get +inf so each is tried once before
    estimates matter (honest init — no fabricated pseudo-rewards)."""
    safe = np.maximum(pull, 1.0)
    return np.where(
        pull > 0.0,
        rew / safe + np.sqrt(1.5 * np.log(t + 2) / safe),
        np.inf,
    )


def bench_cucb(seed: int = 1417, K: int = 8, m: int = 3, T: int = 2500) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, K)
    pull = np.zeros(K)
    rew = np.zeros(K)
    tot = 0.0
    for t in range(T):
        play = np.argsort(-_ucb_scores(rew, pull, t), kind="stable")[:m]
        r = (rng.random(m) < p[play]).astype(float)
        pull[play] += 1
        rew[play] += r
        tot += p[play].sum()
    tot2 = 0.0
    for _t in range(T):
        play = rng.choice(K, m, replace=False)
        tot2 += p[play].sum()
    return {
        "synthetic_cucb_mean_reward": tot / T / m,
        "synthetic_cucb_random_reward": tot2 / T / m,
        "synthetic_cucb_reward_gain": (tot - tot2) / T / m,
        "synthetic_torch_available": 0.0,
    }
