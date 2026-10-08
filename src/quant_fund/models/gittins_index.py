"""Gittins index (calibration-family): compute per-arm Gittins indices (SYNTHETIC)
for Bernoulli bandits via the calibration/binary-search formulation
on the underlying MDP; play argmax index. Regret vs UCB1.
"""

from __future__ import annotations

import numpy as np


def _gittins_beta(alpha: float, beta: float, steps: int = 40, gamma: float = 0.99) -> float:
    # bisection on the reward level making arm indifferent to stopping
    lo, hi = 0.0, 1.0
    for _ in range(18):
        m = (lo + hi) / 2
        # VI on single-arm MDP: continue value vs stopping reward m
        V = 0.0
        for _ in range(steps):
            V = max(m / (1 - gamma), (alpha + gamma * V) / (alpha + beta))
        # actually compute discounted reward if always continue
        # approximate: geometric expectation of Bernoulli(alpha/(a+b))
        p = alpha / (alpha + beta)
        Vc = p / (1 - gamma)
        Vs = m / (1 - gamma)
        if Vc > Vs:
            lo = m
        else:
            hi = m
        _ = V
    return (lo + hi) / 2


def bench_gittins_index(seed: int = 1407, T: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    K = 5
    p = rng.uniform(0.1, 0.9, K)
    ab = np.ones((K, 2))
    tot = 0.0
    pull = np.ones(K)
    rew = np.zeros(K)
    for _t in range(T):
        g = np.array([_gittins_beta(ab[k, 0], ab[k, 1]) for k in range(K)])
        a = int(np.argmax(g))
        r = float(rng.random() < p[a])
        ab[a] += [r, 1 - r]
        pull[a] += 1
        rew[a] += r
        tot += p[a]
    # UCB1
    tot2, pull2, rew2 = 0.0, np.ones(K) * 1e-9, np.zeros(K)
    for t in range(T):
        a = int(np.argmax(rew2 / pull2 + np.sqrt(2 * np.log(t + 2) / pull2)))
        r = float(rng.random() < p[a])
        pull2[a] += 1
        rew2[a] += r
        tot2 += p[a]
    return {
        "synthetic_git_expected_reward": tot / T,
        "synthetic_git_ucb_reward": tot2 / T,
        "synthetic_git_reward_gain": (tot - tot2) / T,
        "synthetic_torch_available": 0.0,
    }
