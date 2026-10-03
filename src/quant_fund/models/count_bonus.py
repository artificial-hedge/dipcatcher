"""MBIE-style count bonus (Strehl & Littman 2008): r+ = beta / sqrt(N(s)).
Harness: tabular Q-learning on the hard-exploration gridworld; metrics
= state coverage + success rate vs an ε-greedy no-bonus baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ex_synth import q_learn, s2i


def bench_count_bonus(seed: int = 2841, beta: float = 0.5) -> dict[str, float]:
    counts: dict[int, int] = {}

    def bonus(s, sp, st, ep, rng):
        i = s2i(sp)
        counts[i] = counts.get(i, 0) + 1
        return beta / np.sqrt(counts[i])

    _, cov, succ = q_learn(bonus, seed=seed)
    _, cov_b, succ_b = q_learn(lambda s, sp, st, ep, rng: 0.0, seed=seed)
    return {
        "synthetic_count_coverage": float(cov),
        "synthetic_baseline_coverage": float(cov_b),
        "synthetic_count_success": float(succ),
        "synthetic_baseline_success": float(succ_b),
        "torch_available": 0.0,
    }
