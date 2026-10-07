"""Stochastic multi-armed bandit canon: UCB1 (Auer, (SYNTHETIC)
Cesa-Bianchi & Fischer 2002), constant-epsilon greedy, and
explore-then-commit. Pseudo-regret is measured on a synthetic
Bernoulli instance and compared across policies.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def _pull(probs: FloatArray, arm: int, rng: np.random.Generator) -> float:
    return float(rng.random() < probs[arm])


def ucb1(probs: FloatArray, steps: int, rng: np.random.Generator) -> float:
    """UCB1: argmax mu_i + sqrt(2 ln t / n_i). Returns pseudo-regret."""
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.size
    counts = np.zeros(k)
    means = np.zeros(k)
    regret = 0.0
    best = float(probs.max())
    for t in range(steps):
        if t < k:
            a = t
        else:
            bonus = np.sqrt(2.0 * np.log(t) / np.maximum(counts, 1.0))
            a = int(np.argmax(means + bonus))
        r = _pull(probs, a, rng)
        counts[a] += 1.0
        means[a] += (r - means[a]) / counts[a]
        regret += best - probs[a]
    return float(regret)


def epsilon_greedy(
    probs: FloatArray,
    steps: int,
    rng: np.random.Generator,
    eps: float = 0.1,
) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.size
    counts = np.zeros(k)
    means = np.zeros(k)
    regret = 0.0
    best = float(probs.max())
    for _ in range(steps):
        if rng.random() < eps:
            a = int(rng.integers(k))
        else:
            a = int(np.argmax(means)) if counts.any() else int(rng.integers(k))
        r = _pull(probs, a, rng)
        counts[a] += 1.0
        means[a] += (r - means[a]) / counts[a]
        regret += best - probs[a]
    return float(regret)


def explore_then_commit(
    probs: FloatArray,
    steps: int,
    rng: np.random.Generator,
    m: int = 20,
) -> float:
    """Pull each arm m times, then commit to the empirical best."""
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.size
    counts = np.zeros(k)
    means = np.zeros(k)
    regret = 0.0
    best = float(probs.max())
    chosen = -1
    for t in range(steps):
        if t < k * m:
            a = t % k
        else:
            if chosen < 0:
                chosen = int(np.argmax(means))
            a = chosen
        r = _pull(probs, a, rng)
        counts[a] += 1.0
        means[a] += (r - means[a]) / counts[a]
        regret += best - probs[a]
    return float(regret)


def _avg_regret(probs: FloatArray, fn, steps: int, runs: int, rng) -> float:
    vals = [fn(probs, steps, rng) for _ in range(runs)]
    return float(np.mean(vals))


def bench_stochastic_bandits(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    k = 10
    probs = np.full(k, 0.5)
    probs[0] = 0.72
    steps, runs = 2000, 8
    u = _avg_regret(probs, ucb1, steps, runs, rng)
    e = _avg_regret(
        probs,
        lambda p, s, r: epsilon_greedy(p, s, r, eps=0.1),
        steps,
        runs,
        rng,
    )
    c = _avg_regret(
        probs,
        lambda p, s, r: explore_then_commit(p, s, r, m=20),
        steps,
        runs,
        rng,
    )
    return {
        "synthetic_ucb1_regret": u,
        "synthetic_eps_greedy_regret": e,
        "synthetic_etc_regret": c,
        "synthetic_max_regret": max(u, e, c),
    }
