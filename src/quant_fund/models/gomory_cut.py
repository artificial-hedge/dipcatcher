"""Gomory fractional cutting-plane ILP solver.

Solves the LP relaxation on the shared tableau (explicit slack columns,
Bland's rule), then appends Gomory fractional cuts from rows whose basic
original variable is non-integral: sum_j frac(t_ij) x_j >= frac(b_i).
Each cut leaves the tableau dual-feasible, so a dual-simplex pivot loop
re-optimizes. Terminates integral or at the cut cap; compared against
the brute-force ILP optimum on the shared fixture.
"""

import numpy as np

from quant_fund.models._ilp_synth import ILP_A, ILP_B, ILP_C, ILP_UB, brute_force_ilp

_EPS = 1e-9


def _pivot(t: np.ndarray, basis: list[int], i: int, j: int) -> None:
    t[i] = t[i] / t[i, j]
    for r in range(t.shape[0]):
        if r != i:
            t[r] = t[r] - t[r, j] * t[i]
    basis[i] = j


def _primal(t: np.ndarray, basis: list[int], cost: np.ndarray) -> None:
    n_v = t.shape[1] - 1
    while True:
        reduced = cost[:n_v] - cost[basis] @ t[:, :n_v]
        j = -1
        for cand in range(n_v):
            if reduced[cand] > _EPS:
                j = cand
                break
        if j < 0:
            return
        col = t[:, j]
        cand_rows = [i for i in range(t.shape[0]) if col[i] > _EPS]
        if not cand_rows:
            raise ValueError("unbounded")
        i = min(cand_rows, key=lambda r: (t[r, -1] / col[r], basis[r]))
        _pivot(t, basis, i, j)


def _dual(t: np.ndarray, basis: list[int], cost: np.ndarray) -> None:
    n_v = t.shape[1] - 1
    while (t[:, -1] < -_EPS).any():
        i = int(np.argmin(t[:, -1]))
        reduced = cost[:n_v] - cost[basis] @ t[:, :n_v]
        neg = [j for j in range(n_v) if t[i, j] < -_EPS]
        if not neg:
            raise ValueError("dual infeasible")
        j = min(neg, key=lambda c: (abs(reduced[c] / t[i, c]), c))
        _pivot(t, basis, i, j)


def _gomory_solve(c: np.ndarray, a: np.ndarray, b: np.ndarray, max_cuts: int = 15):
    m0, n0 = a.shape
    a_full = np.hstack([a.astype(float), np.eye(m0)])
    cost = np.concatenate([c.astype(float), np.zeros(m0 + max_cuts + 1)])
    basis: list[int] = list(range(n0, n0 + m0))
    t = np.hstack([a_full, np.zeros((m0, max_cuts + 1)), b.reshape(-1, 1)])
    _primal(t, basis, cost)
    n_orig = n0
    cuts = 0
    while cuts < max_cuts:
        frac_row = -1
        for i in range(t.shape[0]):
            if basis[i] < n_orig and abs(t[i, -1] - round(t[i, -1])) > 1e-6:
                frac_row = i
                break
        if frac_row < 0:
            break
        n_cols = t.shape[1] - 1
        f = t[frac_row, :n_cols] - np.floor(t[frac_row, :n_cols])
        f0 = t[frac_row, -1] - np.floor(t[frac_row, -1])
        new_row = np.zeros(t.shape[1])
        new_row[:n_cols] = -f
        slack_col = n0 + m0 + cuts
        new_row[slack_col] = 1.0
        new_row[-1] = -f0
        t = np.vstack([t, new_row])
        basis.append(slack_col)
        _dual(t, basis, cost)
        _primal(t, basis, cost)
        cuts += 1
    n_v = t.shape[1] - 1
    x = np.zeros(n_v)
    x[basis] = t[:, -1]
    return x[:n_orig], float(c @ x[:n_orig]), cuts


def bench_gomory_cut(seed: int = 5101) -> dict[str, float]:
    a_full = np.hstack([ILP_A.astype(float), np.eye(2)])
    t = np.hstack([a_full, ILP_B.reshape(-1, 1)])
    basis = [2, 3]
    cost = np.concatenate([ILP_C, np.zeros(4)])
    _primal(t, basis, cost)
    x_lp = np.zeros(4)
    x_lp[basis] = t[:, -1]
    lp_obj = float(ILP_C @ x_lp[:2])
    x, obj, cuts = _gomory_solve(ILP_C, ILP_A, ILP_B)
    truth = brute_force_ilp(ILP_C, ILP_A, ILP_B, ILP_UB)
    return {
        "synthetic_gom_lp_obj": lp_obj,
        "synthetic_gom_obj": obj,
        "synthetic_gom_truth": truth,
        "synthetic_gom_gap": abs(obj - truth),
        "synthetic_gom_cuts": float(cuts),
        "synthetic_gom_integral": float(np.allclose(x, np.round(x), atol=1e-6)),
    }
