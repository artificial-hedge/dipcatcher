"""UGapE (Gabillon et al. 2012) — fixed-budget BAI by adaptive
gap exploration: each round pull the arm minimizing the UCB-gap
confidence index B(i) = max_{j≠i} UCB_j − LCB_i, stopping at the
budget."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.lil_ucb import GaussianBandit

FloatArray = NDArray[np.float64]


def ugape(
    mu: FloatArray,
    rng: np.random.Generator,
    budget: int = 1500,
    sigma: float = 1.0,
) -> tuple[int, int]:
    """UGapE-m: pull the arm minimizing the index gap each round.
    Returns (recommended arm, pulls)."""
    env = GaussianBandit(mu, sigma)
    k = env.k
    for i in range(k):
        env.pull(i, rng)
    t = k
    while t < budget:
        means = env.means()
        t_ns = env.counts
        rad = np.sqrt(2.0 * sigma**2 * np.log(4.0 * t * t / 1.0) / t_ns)
        ucb = means + rad
        lcb = means - rad
        # B(i) = max_{j≠i} UCB_j − LCB_i ; pull argmin over all arms
        best_gap = np.inf
        choice = 0
        for i in range(k):
            others = np.delete(ucb, i)
            gap = float(others.max() - lcb[i])
            if gap < best_gap:
                best_gap, choice = gap, i
        # pull the gap-minimizer AND its UCB challenger (UGapE-m)
        masked = ucb.copy()
        masked[choice] = -np.inf
        challenger = int(np.argmax(masked))
        env.pull(choice, rng)
        env.pull(challenger, rng)
        t += 2
    means = env.means()
    return int(np.argmax(means)), t


def bench_ugape(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: UGapE identifies the best arm on a 6-arm instance
    inside budget."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    mu = np.array([0.0, 0.25, 0.1, 0.2, 0.05, 0.15])
    best, pulls = ugape(mu, rng, budget=1500)
    out["synthetic_ugape_correct"] = float(best == 1)
    out["synthetic_ugape_pulls"] = float(pulls)
    out["synthetic_ugape_budget_ok"] = float(pulls == 1500)
    # closer means
    mu2 = np.array([0.0, 0.4, 0.35])
    best2, pulls2 = ugape(mu2, rng, budget=8000)
    out["synthetic_ugape_close_correct"] = float(best2 == 1)
    out["synthetic_ugape_close_pulls"] = float(pulls2)
    return out


if __name__ == "__main__":
    print(bench_ugape())
