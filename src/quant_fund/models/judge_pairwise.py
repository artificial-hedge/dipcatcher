"""LLM-as-judge — pairwise scoring (Zheng et al. 2023, MT-Bench) (SYNTHETIC).

A judge model scores pairwise comparisons between candidate responses
(noisy estimate of true quality); agreement rate with the oracle
ranking and Elo-sorted order recovery quantify judge fidelity.
"""

from __future__ import annotations

import numpy as np


def bench_judge_pairwise(
    seed: int = 401,
    n_items: int = 40,
    noise: float = 0.15,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0, 1, n_items)
    # pairwise comparisons by judge: argmax of noisy estimates
    agree = 0
    total = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            ji = quality[i] + rng.normal(0, noise)
            jj = quality[j] + rng.normal(0, noise)
            agree += int((ji > jj) == (quality[i] > quality[j]))
            total += 1
    # Elo recovery: win-rate ordering vs true ordering (spearman)
    wins = np.zeros(n_items)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            ji = quality[i] + rng.normal(0, noise)
            jj = quality[j] + rng.normal(0, noise)
            if ji > jj:
                wins[i] += 1
            else:
                wins[j] += 1
    order_judge = wins.argsort()
    order_true = quality.argsort()
    rho = float(
        np.corrcoef(
            np.argsort(np.argsort(order_judge)),
            np.argsort(np.argsort(order_true)),
        )[0, 1]
    )
    return {
        "synthetic_judge_agreement": agree / total,
        "synthetic_judge_rank_rho": rho,
        "synthetic_judge_random_agree": 0.5,
    }
