"""Corruption-tolerant bandit (Bogunovic et al. 2021) — trimmed-mean /
median-based UCB robust to adversarial reward corruption in c fraction
of pulls. Regret vs mean-UCB under corruption.
"""

from __future__ import annotations

import numpy as np


def bench_corrupt_bandit(
    seed: int = 1423, K: int = 5, T: int = 3000, corrupt: float = 0.15
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.8, K)
    # robust bandit: median-of-rewards per arm + UCB on count
    obs: list[list[float]] = [[] for _ in range(K)]
    tot = 0.0
    for t in range(T):
        med = np.array([np.median(o) if o else 1.0 for o in obs])
        cnt = np.array([max(len(o), 1) for o in obs])
        a = int(np.argmax(med + np.sqrt(2 * np.log(t + 2) / cnt)))
        r = float(rng.random() < p[a])
        if rng.random() < corrupt:
            r = 1.0 - r  # adversarial flip
        obs[a].append(r)
        tot += p[a]
    # mean-UCB naive under corruption
    pull = np.ones(K)
    rew = np.zeros(K)
    tot2 = 0.0
    for t in range(T):
        a = int(np.argmax(rew / pull + np.sqrt(2 * np.log(t + 2) / pull)))
        r = float(rng.random() < p[a])
        if rng.random() < corrupt:
            r = 1.0 - r
        pull[a] += 1
        rew[a] += r
        tot2 += p[a]
    return {
        "synthetic_cb_robust_reward": tot / T,
        "synthetic_cb_naive_reward": tot2 / T,
        "synthetic_cb_reward_gain": (tot - tot2) / T,
        "synthetic_torch_available": 0.0,
    }
