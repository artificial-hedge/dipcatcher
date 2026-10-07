"""0/1 knapsack exact DP vs greedy density heuristic — optimal-value
recovery and greedy gap on random instances.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sched_synth import knapsack


def _dp(w: np.ndarray, v: np.ndarray, cap: float) -> float:
    n = len(w)
    C = int(cap)
    dp = np.zeros(C + 1)
    for i in range(n):
        wi, vi = int(w[i]), v[i]
        dp[wi:] = np.maximum(dp[wi:], dp[: C + 1 - wi] + vi)
    return float(dp[C])


def bench_knapsack_dp(seed: int = 3053) -> dict[str, float]:
    w, v, cap = knapsack(seed)
    opt = _dp(w, v, cap)
    # greedy by value/weight density
    order = np.argsort(-v / w)
    load = val = 0.0
    for i in order:
        if load + w[i] <= cap:
            load += w[i]
            val += v[i]
    return {
        "synthetic_ks_opt": opt,
        "synthetic_ks_greedy": float(val),
        "synthetic_ks_gap": float(opt - val),
        "synthetic_ks_greedy_ratio": float(val / opt),
        "synthetic_torch_available": 0.0,
    }
