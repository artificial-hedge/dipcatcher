"""Gittins index (calibration-family): compute per-arm Gittins indices (SYNTHETIC)
for Bernoulli bandits via the calibration/binary-search formulation
on the underlying MDP; play argmax index. Regret vs UCB1.
"""

from __future__ import annotations

import numpy as np


def _gittins_beta(
    alpha: float,
    beta: float,
    gamma: float = 0.99,
    horizon: int = 40,
    iters: int = 18,
) -> float:
    """Gittins index of a Beta(alpha, beta) Bernoulli arm.

    Calibration (fair-charge) formulation: the arm with charge ``m``
    pays ``m`` per period on stopping, so the value from state (a,b)
    with ``p = a/(a+b)`` is

        V(a,b) = max(m/(1-gamma),
                     p + gamma * [p V(a+1,b) + (1-p) V(a,b+1)])

    and the index is the largest ``m`` where continuing is optimal —
    bisected over [0, 1] with backward induction truncated at
    ``horizon`` (tail value max(stop, p/(1-gamma))).
    """
    ia = np.arange(horizon + 1)[:, None]
    jb = np.arange(horizon + 1)[None, :]
    p = (alpha + ia) / (alpha + beta + ia + jb)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        stop = m / (1.0 - gamma)
        # terminal: remaining steps exhausted — stop, or keep playing
        # under the static-p approximation
        v = np.maximum(stop, p / (1.0 - gamma))
        for _h in range(horizon):
            vp = np.pad(v[1:, :], ((0, 1), (0, 0)), constant_values=stop)
            vq = np.pad(v[:, 1:], ((0, 0), (0, 1)), constant_values=stop)
            v = np.maximum(stop, p + gamma * (p * vp + (1.0 - p) * vq))
        if v[0, 0] > stop:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def bench_gittins_index(seed: int = 1407, T: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    K = 5
    p = rng.uniform(0.1, 0.9, K)
    ab = np.ones((K, 2))
    tot = 0.0
    pull = np.ones(K)
    rew = np.zeros(K)
    # only the pulled arm's posterior moves — recompute its index,
    # keep the others cached
    g = np.array([_gittins_beta(ab[k, 0], ab[k, 1]) for k in range(K)])
    for _t in range(T):
        a = int(np.argmax(g))
        r = float(rng.random() < p[a])
        ab[a] += [r, 1 - r]
        pull[a] += 1
        rew[a] += r
        tot += p[a]
        g[a] = _gittins_beta(ab[a, 0], ab[a, 1])
    # UCB1
    tot2, pull2, rew2 = 0.0, np.ones(K) * 1e-9, np.zeros(K)
    for t in range(T):
        a = int(np.argmax(rew2 / pull2 + np.sqrt(2 * np.log(t + 2) / pull2)))
        r = float(rng.random() < p[a])
        pull2[a] += 1
        rew2[a] += r
        tot2 += p[a]
    out = {
        "synthetic_git_expected_reward": tot / T,
        "synthetic_git_ucb_reward": tot2 / T,
        "synthetic_git_reward_gain": (tot - tot2) / T,
        "synthetic_torch_available": 0.0,
    }
    # the index policy must concentrate pulls on the best arm: its
    # per-step expected reward approaches max(p), and it must at least
    # match UCB1
    if out["synthetic_git_expected_reward"] < float(p.max()) - 0.05:
        raise ValueError(
            f"gittins policy off: {out['synthetic_git_expected_reward']:.3f} vs best {p.max():.3f}"
        )
    if out["synthetic_git_reward_gain"] < -0.02:
        raise ValueError(f"gittins loses to ucb1: {out['synthetic_git_reward_gain']:.3f}")
    return out
