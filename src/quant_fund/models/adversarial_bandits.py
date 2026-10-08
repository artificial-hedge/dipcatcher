"""Adversarial bandit canon: EXP3 (Auer, Cesa-Bianchi, (SYNTHETIC)
Freund & Schapire 2002) with importance-weighted reward
estimates, against a uniform-play baseline and a full-info
Hedge upper bound, on a synthetic rotating-best-arm
adversarial reward table.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def exp3(
    rewards: FloatArray,
    rng: np.random.Generator,
    gamma: float = 0.07,
) -> float:
    """EXP3 on a (T, K) reward table; returns pseudo-regret."""
    rewards = np.asarray(rewards, dtype=np.float64)
    t_steps, k = rewards.shape
    w = np.ones(k)
    regret = 0.0
    for t in range(t_steps):
        p = w / w.sum()
        p = (1.0 - gamma) * p + gamma / k
        a = int(rng.choice(k, p=p))
        r = rewards[t, a]
        x_hat = r / p[a]
        w[a] *= np.exp(gamma * x_hat / k)
        regret += float(rewards[t].max() - r)
    return float(regret)


def hedge(rewards: FloatArray, eta: float = 0.5) -> float:
    """Full-info Hedge (best-response benchmark); pseudo-regret."""
    rewards = np.asarray(rewards, dtype=np.float64)
    t_steps, k = rewards.shape
    w = np.ones(k)
    regret = 0.0
    for t in range(t_steps):
        p = w / w.sum()
        got = float(p @ rewards[t])
        w *= np.exp(eta * rewards[t])
        w = np.maximum(w, 1e-30)
        regret += float(rewards[t].max() - got)
    return float(regret)


def _adversarial_table(t_steps: int, k: int, rng) -> FloatArray:
    """Adversarial rewards: arm 0 stays best but all levels vary
    deterministically so no stochastic iid structure is available."""
    tab = np.zeros((t_steps, k))
    for t in range(t_steps):
        tab[t, 0] = 0.55 + 0.3 * np.sin(0.37 * t) + 0.1 * np.sin(0.031 * t * t)
        for a in range(1, k):
            tab[t, a] = 0.15 + 0.25 * np.abs(np.sin(0.23 * t + a)) * (a / k)
    return np.clip(tab, 0.0, 1.0)


def bench_adversarial_bandits(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    t_steps, k = 2000, 5
    tab = _adversarial_table(t_steps, k, rng)
    runs = 6
    e = float(np.mean([exp3(tab, rng) for _ in range(runs)]))
    h = hedge(tab)
    uniform = float(np.sum(tab.max(axis=1) - tab.mean(axis=1)))
    return {
        "synthetic_exp3_regret": e,
        "synthetic_hedge_regret": h,
        "synthetic_uniform_regret": uniform,
        "synthetic_exp3_beats_uniform": 1.0 if e < uniform else 0.0,
    }
