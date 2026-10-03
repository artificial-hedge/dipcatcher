"""Branch-and-cut on the shared 2-variable ILP.

Depth-first branch-and-bound over fractional LP solutions (bounds from
scipy linprog since bound rows can carry negative RHS), vs the same
search with a Gomory cut closure applied at the root node first. Bench
reports the two node counts, the shared optimum vs brute force, and the
node savings from the cut closure.
"""

import numpy as np
from scipy.optimize import linprog

from quant_fund.models._ilp_synth import ILP_A, ILP_B, ILP_C, ILP_UB, brute_force_ilp
from quant_fund.models.gomory_cut import _gomory_solve


def _lp(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray | None, float]:
    res = linprog(-c, A_ub=a, b_ub=b, bounds=(0, None), method="highs")
    if not res.success:
        return None, -np.inf
    return res.x, float(-res.fun)


def _bnb(c: np.ndarray, a: np.ndarray, b: np.ndarray, use_root_cuts: bool):
    stack = [(a.copy(), b.copy())]
    if use_root_cuts:
        x0, obj0, _ = _gomory_solve(c, a, b)
        if np.allclose(x0, np.round(x0), atol=1e-6):
            return obj0, 1, x0
    nodes = 0
    best_obj = -np.inf
    best_x = None
    while stack and nodes < 5000:
        aa, bb = stack.pop()
        nodes += 1
        x, obj = _lp(c, aa, bb)
        if x is None or obj <= best_obj + 1e-9:
            continue
        frac = [j for j in range(len(c)) if abs(x[j] - round(x[j])) > 1e-6]
        if not frac:
            best_obj, best_x = obj, x.copy()
            continue
        j = frac[0]
        a1 = np.vstack([aa, np.eye(len(c))[j]])
        b1 = np.append(bb, np.floor(x[j]))
        stack.append((a1, b1))
        a2 = np.vstack([aa, -np.eye(len(c))[j]])
        b2 = np.append(bb, -np.ceil(x[j]))
        stack.append((a2, b2))
    return best_obj, nodes, best_x


def bench_branch_and_cut(seed: int = 5109) -> dict[str, float]:
    obj_plain, n_plain, _ = _bnb(ILP_C, ILP_A, ILP_B, False)
    obj_cut, n_cut, _ = _bnb(ILP_C, ILP_A, ILP_B, True)
    truth = brute_force_ilp(ILP_C, ILP_A, ILP_B, ILP_UB)
    return {
        "synthetic_bnc_obj": obj_plain,
        "synthetic_bnc_nodes": float(n_plain),
        "synthetic_bnc_nodes_cut": float(n_cut),
        "synthetic_bnc_truth": truth,
        "synthetic_bnc_gap": abs(obj_plain - truth),
        "synthetic_bnc_gap_cut": abs(obj_cut - truth),
        "synthetic_bnc_nodes_saved": float(n_plain - n_cut),
    }
