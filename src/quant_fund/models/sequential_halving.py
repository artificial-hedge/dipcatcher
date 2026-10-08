"""Sequential Halving (Karnin et al. 2013) — fixed-budget BAI: (SYNTHETIC)
split the budget into ⌈log₂K⌉ rounds, sample each surviving arm
evenly, discard the worst half each round."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.lil_ucb import GaussianBandit

FloatArray = NDArray[np.float64]


def sequential_halving(
    mu: FloatArray,
    rng: np.random.Generator,
    budget: int = 2000,
    sigma: float = 1.0,
) -> tuple[int, int]:
    """SH over ⌈log₂ K⌉ rounds. Returns (recommended arm, pulls)."""
    env = GaussianBandit(mu, sigma)
    alive = list(range(env.k))
    n_rounds = int(np.ceil(np.log2(env.k)))
    per_round = budget // n_rounds
    pulls = 0
    while len(alive) > 1 and pulls < budget:
        each = max(1, per_round // len(alive))
        for i in alive:
            for _ in range(each):
                env.pull(i, rng)
                pulls += 1
        means = env.means()
        alive.sort(key=lambda i: -means[i])
        alive = alive[: max(1, len(alive) // 2)]
    return alive[0], pulls


def bench_sequential_halving(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: SH recovers the best arm on 8-arm instance and
    uses the full budget."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    mu = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25])
    best, pulls = sequential_halving(mu, rng, budget=3000)
    out["synthetic_sh_correct"] = float(best == 5)
    out["synthetic_sh_pulls"] = float(pulls)
    out["synthetic_sh_budget_ok"] = float(2000 <= pulls <= 3000)
    # sparse instance: only arm 6 good
    mu2 = np.zeros(8)
    mu2[6] = 0.3
    best2, _ = sequential_halving(mu2, rng, budget=3000)
    out["synthetic_sh_sparse_correct"] = float(best2 == 6)
    return out


if __name__ == "__main__":
    print(bench_sequential_halving())
