"""Top-Two Thompson Sampling (Russo 2016) — fixed-confidence BAI:
draw posterior means, take the leader with prob β else resample
until a distinct leader appears and pull it; stop on the Chernoff
generalized-likelihood threshold."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.lil_ucb import GaussianBandit

FloatArray = NDArray[np.float64]


def ttts(
    mu: FloatArray,
    rng: np.random.Generator,
    delta: float = 0.1,
    sigma: float = 1.0,
    beta: float = 0.5,
    max_pulls: int = 60000,
) -> tuple[int, int]:
    """Gaussian TTTS with N(μ̂_i, σ²/T_i) posteriors and the GLR
    stopping rule min_{i≠*} T_i T_* (μ̂_*−μ̂_i)²/(2σ²(T_i+T_*)) > β_δ."""
    env = GaussianBandit(mu, sigma)
    k = env.k
    for i in range(k):
        env.pull(i, rng)
    t = k
    while t < max_pulls:
        means = env.means()
        t_ns = env.counts
        # GLR statistic vs the current leader
        star = int(np.argmax(means))
        z = np.inf
        for i in range(k):
            if i == star:
                continue
            num = t_ns[i] * t_ns[star] * (means[star] - means[i]) ** 2
            den = 2.0 * sigma**2 * (t_ns[i] + t_ns[star])
            z = min(z, num / den)
        if z > np.log((np.log(t) + 1.0) / delta):
            return star, t
        # Thompson sample → leader; β-bias else challenger
        theta = rng.normal(means, sigma / np.sqrt(t_ns))
        if rng.random() < beta:
            pull = int(np.argmax(theta))
        else:
            pull = star
            for _ in range(60):
                theta = rng.normal(means, sigma / np.sqrt(t_ns))
                cand = int(np.argmax(theta))
                if cand != star:
                    pull = cand
                    break
        env.pull(pull, rng)
        t += 1
    return int(np.argmax(env.means())), t


def bench_ttts(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: TTTS identifies the best arm and concentrates
    pulls on the top two contenders."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    mu = np.array([0.0, 0.3, 0.1, 0.25])
    best, pulls = ttts(mu, rng, delta=0.1)
    out["synthetic_ttts_correct"] = float(best == 1)
    out["synthetic_ttts_pulls"] = float(pulls)
    out["synthetic_ttts_budget_ok"] = float(pulls < 60000)
    mu2 = np.array([0.0, 0.5, 0.4, 0.2])
    best2, _ = ttts(mu2, rng, delta=0.1)
    out["synthetic_ttts_second"] = float(best2 == 1)
    return out


if __name__ == "__main__":
    print(bench_ttts())
