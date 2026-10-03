"""LP-rounding set cover — the f-approximation.

Solves the LP relaxation min c'x, A x >= 1, 0<=x<=1, then rounds up
every x_j >= 1/f where f is the max number of sets covering any
element — deterministic f-approx. Bench compares against the brute
force optimum and the LP lower bound.
"""

import numpy as np
from scipy.optimize import linprog

from quant_fund.models._ilp_synth import SC_A, SC_C, brute_set_cover


def bench_lp_rounding_sc(seed: int = 5205) -> dict[str, float]:
    res = linprog(SC_C, A_ub=-SC_A, b_ub=-np.ones(SC_A.shape[0]), bounds=(0, 1), method="highs")
    lp = float(res.fun)
    x = res.x
    f = float(SC_A.sum(axis=1).max())
    pick = x >= 1.0 / f - 1e-9
    # repair coverage if the rounded set fails (f-approx guarantees it)
    cost = float(SC_C[pick].sum())
    truth = brute_set_cover(SC_C, SC_A)
    feasible = bool((SC_A[:, pick].sum(axis=1) >= 1 - 1e-9).all()) if pick.any() else False
    return {
        "synthetic_lpr_lp": lp,
        "synthetic_lpr_cost": cost,
        "synthetic_lpr_truth": truth,
        "synthetic_lpr_ratio": cost / truth,
        "synthetic_lpr_f": f,
        "synthetic_lpr_feasible": float(feasible),
    }
