"""RIDE (Raileanu & Rocktaschel 2020): intrinsic reward = |f(s') - f(s)|
on learned/fixed features — encourages actions that change the state.
Numpy fixed random-projection features + forward-model variant.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ex_synth import q_learn, state_feat


def bench_ride_explore(seed: int = 2861) -> dict[str, float]:
    rng0 = np.random.default_rng(seed)
    proj = rng0.normal(0, 1, (6, 16))
    lifecnt: dict[int, int] = {}

    def f(s):
        return np.tanh(state_feat(s) @ proj)

    def bonus(s, sp, st, ep, rng):
        i = sp[0] * 7 + sp[1]
        lifecnt[i] = lifecnt.get(i, 0) + 1
        delta = float(np.linalg.norm(f(sp) - f(s)))
        return delta / np.sqrt(max(lifecnt[i], 1))

    _, cov, succ = q_learn(bonus, seed=seed)
    _, cov_b, succ_b = q_learn(lambda s, sp, st, ep, rng: 0.0, seed=seed)
    return {
        "synthetic_ride_coverage": float(cov),
        "synthetic_baseline_coverage": float(cov_b),
        "synthetic_ride_success": float(succ),
        "synthetic_baseline_success": float(succ_b),
        "torch_available": 0.0,
    }
