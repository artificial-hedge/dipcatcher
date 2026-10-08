"""FPTAS for the 0/1 knapsack (SYNTHETIC).

Values scaled down by eps * max_v / n, then exact DP on integer values
gives a (1-eps)-approximation in poly(n/eps) — the classic Ibarra-Kim
scheme. Bench compares the FPTAS value against the brute-force optimum
and reports the true approximation achieved.
"""

import numpy as np

from quant_fund.models._approx_synth import KS_CAP, KS_V, KS_W, brute_knapsack


def _fptas(eps: float = 0.1) -> tuple[float, int]:
    n = len(KS_V)
    scale = eps * float(KS_V.max()) / n
    v_scaled = np.floor(KS_V / scale).astype(int)
    v_max = int(v_scaled.max() * n)
    dp = np.full(v_max + 1, np.inf)
    dp[0] = 0.0
    for i in range(n):
        vi, wi = v_scaled[i], KS_W[i]
        for val in range(v_max, vi - 1, -1):
            if dp[val - vi] + wi < dp[val]:
                dp[val] = dp[val - vi] + wi
    best_val = max(v for v in range(v_max + 1) if dp[v] <= KS_CAP + 1e-9)
    # recover items
    items = 0
    dp2 = np.full(v_max + 1, np.inf)
    dp2[0] = 0.0
    take = np.zeros((n, v_max + 1), dtype=bool)
    for i in range(n):
        for v in range(v_max, v_scaled[i] - 1, -1):
            if dp2[v - v_scaled[i]] + KS_W[i] < dp2[v]:
                dp2[v] = dp2[v - v_scaled[i]] + KS_W[i]
                take[i, v] = True
    real = 0.0
    v = best_val
    for i in range(n - 1, -1, -1):
        if take[i, v]:
            real += KS_V[i]
            items += 1
            v -= v_scaled[i]
    return real, items


def bench_fptas_knapsack(seed: int = 5207) -> dict[str, float]:
    approx, items = _fptas(0.1)
    truth = brute_knapsack(KS_V, KS_W, KS_CAP)
    return {
        "synthetic_fptas_val": approx,
        "synthetic_fptas_items": float(items),
        "synthetic_fptas_truth": truth,
        "synthetic_fptas_ratio": approx / truth,
    }
