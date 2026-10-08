"""Gilmore-Gomory column generation for the cutting-stock problem (SYNTHETIC).

Restricted master: min sum_p x_p over generated patterns with
sum_p a_ip x_p >= demand_i. Pricing: knapsack subproblem on duals via
DP (max sum_i pi_i a_i subject to sum_i w_i a_i <= stock). Iterates
until no column prices out; bench reports the LP bound vs the rounded
integer solution and vs the trivial full-width baseline.
"""

import numpy as np
from scipy.optimize import linprog

from quant_fund.models._ilp_synth import CS_DEMAND, CS_W, CS_WIDTHS


def _knapsack_price(pi: np.ndarray, widths: np.ndarray, w: float) -> np.ndarray | None:
    """Max pi.a s.t. w'a <= W; returns pattern a or None if z<=1+eps."""
    w_int = np.round(widths).astype(int)
    w_max = int(round(w))
    dp = np.zeros(w_max + 1)
    pat = np.zeros((w_max + 1, len(widths)), dtype=int)
    for cap in range(1, w_max + 1):
        for i, wi in enumerate(w_int):
            if wi <= cap and dp[cap - wi] + pi[i] > dp[cap]:
                dp[cap] = dp[cap - wi] + pi[i]
                pat[cap] = pat[cap - wi]
                pat[cap, i] += 1
    if dp[w_max] > 1.0 + 1e-9:
        return np.asarray(pat[w_max], dtype=float)
    return None


def _colgen() -> tuple[float, int]:
    patterns = [np.eye(len(CS_WIDTHS))[i] * (CS_W // CS_WIDTHS[i]) for i in range(len(CS_WIDTHS))]
    for _ in range(60):
        a = np.stack(patterns).T  # items x patterns
        # master: min 1'x s.t. A x >= d, x >= 0
        res = linprog(
            np.ones(a.shape[1]),
            A_ub=-a,
            b_ub=-CS_DEMAND.astype(float),
            bounds=(0, None),
            method="highs",
        )
        # dual prices on the demand rows come back negated for A_ub
        pi = np.maximum(0.0, -np.asarray(res.ineqlin.marginals))
        new_col = _knapsack_price(pi, CS_WIDTHS, CS_W)
        if new_col is None:
            return float(res.fun), len(patterns)
        patterns.append(new_col)
    return float(res.fun), len(patterns)


def bench_column_generation(seed: int = 5103) -> dict[str, float]:
    lp_opt, n_pat = _colgen()
    # simple integer heuristic: full-width-per-item count
    naive = float(np.ceil(CS_DEMAND / (CS_W // CS_WIDTHS)).sum())
    return {
        "synthetic_cg_lp_bound": lp_opt,
        "synthetic_cg_patterns": float(n_pat),
        "synthetic_cg_naive_rolls": naive,
        "synthetic_cg_bound_vs_naive": float(naive - np.ceil(lp_opt)),
    }
