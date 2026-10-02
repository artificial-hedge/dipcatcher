"""Non-stationary bandit canon: sliding-window UCB and
discounted UCB (Garivier & Moulines 2011) on a piecewise-
stationary Bernoulli environment whose optimal arm rotates,
against a stationary UCB1 baseline that cannot track changes.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def _regret(probs_seq: FloatArray, picks: list[int]) -> float:
    best = probs_seq.max(axis=1)
    got = np.array([probs_seq[t, a] for t, a in enumerate(picks)])
    return float(np.sum(best - got))


def sw_ucb(
    probs_seq: FloatArray,
    rng: np.random.Generator,
    window: int = 400,
) -> float:
    """Sliding-window UCB: statistics over the last W plays only."""
    t_steps, k = probs_seq.shape
    pulls: list[list[tuple[int, float]]] = [[] for _ in range(k)]
    picks: list[int] = []
    for t in range(t_steps):
        scores = []
        for a in range(k):
            recent = [(ts, r) for ts, r in pulls[a] if ts > t - window]
            if not recent:
                scores.append(np.inf)
                continue
            n_a = len(recent)
            mu = float(np.mean([r for _, r in recent]))
            xi = np.sqrt(0.5 * np.log(min(t + 1, window * 4)) / n_a)
            scores.append(mu + xi)
        a = int(np.argmax(scores))
        r = float(rng.random() < probs_seq[t, a])
        pulls[a].append((t, r))
        picks.append(a)
    return _regret(probs_seq, picks)


def d_ucb(
    probs_seq: FloatArray,
    rng: np.random.Generator,
    gamma_disc: float = 0.997,
) -> float:
    """Discounted UCB: exponentially weighted counts and means."""
    t_steps, k = probs_seq.shape
    counts = np.zeros(k)
    rewards = np.zeros(k)
    picks: list[int] = []
    for t in range(t_steps):
        if t < k:
            a = t
        else:
            n_total = max(float(counts.sum()), 1e-9)
            scores = rewards / np.maximum(counts, 1e-9) + np.sqrt(
                2.0 * np.log(n_total) / np.maximum(counts, 1e-9)
            )
            a = int(np.argmax(scores))
        r = float(rng.random() < probs_seq[t, a])
        counts = counts * gamma_disc
        rewards = rewards * gamma_disc
        counts[a] += 1.0
        rewards[a] += r
        picks.append(a)
    return _regret(probs_seq, picks)


def _ucb1_baseline(probs_seq: FloatArray, rng: np.random.Generator) -> float:
    """Stationary UCB1 (expected to fail on drift)."""
    t_steps, k = probs_seq.shape
    counts = np.zeros(k)
    means = np.zeros(k)
    picks: list[int] = []
    for t in range(t_steps):
        if t < k:
            a = t
        else:
            bonus = np.sqrt(2.0 * np.log(t) / np.maximum(counts, 1.0))
            a = int(np.argmax(means + bonus))
        r = float(rng.random() < probs_seq[t, a])
        counts[a] += 1.0
        means[a] += (r - means[a]) / counts[a]
        picks.append(a)
    return _regret(probs_seq, picks)


def _piecewise(t_steps: int, k: int, phase: int) -> FloatArray:
    """Best arm rotates each phase; suboptimal arms fixed at 0.4."""
    seq = np.full((t_steps, k), 0.4)
    n_phases = t_steps // phase
    for ph in range(n_phases):
        best = ph % k
        seq[ph * phase : (ph + 1) * phase, best] = 0.75
    return seq


def bench_nonstationary_bandits(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    t_steps, k, phase = 3000, 5, 500
    seq = _piecewise(t_steps, k, phase)
    runs = 4
    sw = float(np.mean([sw_ucb(seq, rng, window=150) for _ in range(runs)]))
    du = float(np.mean([d_ucb(seq, rng, gamma_disc=0.99) for _ in range(runs)]))
    u = float(np.mean([_ucb1_baseline(seq, rng) for _ in range(runs)]))
    return {
        "synthetic_sw_ucb_regret": sw,
        "synthetic_d_ucb_regret": du,
        "synthetic_ucb1_regret": u,
    }
