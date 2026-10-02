"""LUCB / lil'UCB (Jamieson et al. 2014) — anytime-confidence
pure-exploration: pull argmax UCB and the best challenger by LCB,
stop when the leader's LCB clears every challenger's UCB.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class GaussianBandit:
    """K-arm Gaussian bandit with unit-variance rewards."""

    def __init__(self, mu: FloatArray, sigma: float = 1.0) -> None:
        self.mu = np.asarray(mu, dtype=np.float64)
        self.k = len(self.mu)
        self.sigma = sigma
        self.counts = np.zeros(self.k)
        self.sums = np.zeros(self.k)

    def pull(self, i: int, rng: np.random.Generator) -> float:
        r = float(rng.normal(self.mu[i], self.sigma))
        self.counts[i] += 1
        self.sums[i] += r
        return r

    def means(self) -> FloatArray:
        out: FloatArray = np.asarray(self.sums / np.maximum(self.counts, 1))
        return out


def lil_ucb(
    mu: FloatArray,
    rng: np.random.Generator,
    delta: float = 0.1,
    sigma: float = 1.0,
    max_pulls: int = 20000,
) -> tuple[int, int]:
    """LUCB with anytime radius sqrt(2σ² log((log₂(t)+1)·c/δ)/t).
    Returns (recommended arm, pulls used)."""
    env = GaussianBandit(mu, sigma)
    k = env.k
    for i in range(k):
        env.pull(i, rng)
    t = k
    while t < max_pulls:
        means = env.means()
        t_ns = env.counts
        rad = np.sqrt(
            2.0 * sigma**2 * np.log((np.log2(np.maximum(t_ns, 2)) + 1.0) * 4.0 * k / delta) / t_ns
        )
        ucb = means + rad
        lcb = means - rad
        best = int(np.argmax(means))
        challenger = int(np.argmax(lcb + np.where(np.arange(k) == best, -np.inf, 0)))
        if lcb[best] >= ucb[challenger]:
            return best, t
        env.pull(int(np.argmax(ucb)), rng)
        env.pull(challenger, rng)
        t += 2
    return int(np.argmax(env.means())), t


def bench_lil_ucb(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: 5-arm Gaussian bandit — LUCB identifies the best
    arm within budget and its sample count stays O(K/Δ² log)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    mu = np.array([0.0, 0.3, 0.5, 0.2, 0.1])
    best, pulls = lil_ucb(mu, rng, delta=0.1)
    out["synthetic_lilucb_correct"] = float(best == 2)
    out["synthetic_lilucb_pulls"] = float(pulls)
    out["synthetic_lilucb_budget_ok"] = float(pulls < 20000)
    # harder instance: Δ=0.2
    mu2 = np.array([0.0, 0.2, 0.15, 0.1])
    best2, pulls2 = lil_ucb(mu2, rng, delta=0.1)
    out["synthetic_lilucb_hard_correct"] = float(best2 == 1)
    out["synthetic_lilucb_hard_pulls"] = float(pulls2)
    return out


if __name__ == "__main__":
    print(bench_lil_ucb())
