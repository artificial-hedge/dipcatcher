"""CUCB (Chen et al. 2013) — combinatorial UCB semi-bandit: pick top-m
arms each round by UCB on per-arm mean estimates; observe all chosen.
Expected reward vs random top-m.
"""

from __future__ import annotations

import numpy as np


def bench_cucb(seed: int = 1417, K: int = 8, m: int = 3, T: int = 2500) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, K)
    pull = np.ones(K)
    rew = rng.random(K) * 0.01
    tot = 0.0
    for t in range(T):
        ucb = rew / pull + np.sqrt(1.5 * np.log(t + 2) / pull)
        play = np.argsort(-ucb)[:m]
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
