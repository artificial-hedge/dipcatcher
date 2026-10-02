"""Median Elimination (Even-Dar et al. 2006) — (ε,δ)-PAC BAI:
eliminate the worst half of the candidate set each epoch until one
arm remains; sample counts grow as O((K/ε²) log(1/δ))."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.lil_ucb import GaussianBandit

FloatArray = NDArray[np.float64]


def median_elimination(
    mu: FloatArray,
    rng: np.random.Generator,
    eps: float = 0.15,
    delta: float = 0.1,
    sigma: float = 1.0,
) -> tuple[int, int]:
    """Median elimination: epoch ℓ samples each surviving arm
    (4σ²/ε_ℓ²)·log(3/δ_ℓ) times, keeps the top-median half with
    ε_ℓ = 5ε/8·(4/5)^ℓ, δ_ℓ = δ/2^ℓ."""
    env = GaussianBandit(mu, sigma)
    alive = list(range(env.k))
    eps_l, del_l, pulls = eps / 2.0, delta / 2.0, 0
    while len(alive) > 1:
        m = len(alive)
        n_i = int(np.ceil(4.0 * sigma**2 * np.log(3.0 / del_l) / eps_l**2))
        sums = np.zeros(m)
        for j, i in enumerate(alive):
            for _ in range(n_i):
                sums[j] += env.pull(i, rng)
                pulls += 1
        means_i = sums / n_i
        med = float(np.median(means_i))
        alive = [i for j, i in enumerate(alive) if means_i[j] >= med]
        eps_l *= 0.75
        del_l *= 0.5
        if pulls > 2_000_000:
            break
    return alive[0], pulls


def bench_median_elim(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: median elimination returns an ε-best arm (not
    necessarily the exact argmax — PAC guarantee)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    mu = np.array([0.0, 0.1, 0.3, 0.05, 0.28])
    best, pulls = median_elimination(mu, rng, eps=0.15, delta=0.1)
    out["synthetic_me_eps_best"] = float(mu[best] >= 0.3 - 0.15)
    out["synthetic_me_pulls"] = float(pulls)
    out["synthetic_me_exact"] = float(best == 2)
    return out


if __name__ == "__main__":
    print(bench_median_elim())
