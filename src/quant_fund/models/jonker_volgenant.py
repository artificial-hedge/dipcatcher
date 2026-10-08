"""Jonker–Volgenant shortest-augmenting-path linear assignment (SYNTHETIC).

Canonical reference: Jonker & Volgenant (1987); Crouse (2016) for the
Dijkstra-form dual updates. Solves  min Σ c[i, π(i)]  on rectangular
cost matrices via column reduction + augmenting paths — O(n³).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_BIG = 1e18


def _augment(
    c: FloatArray,
    u: FloatArray,
    v: FloatArray,
    row2col: np.ndarray,
    col2row: np.ndarray,
    free_row: int,
) -> None:
    """Dijkstra shortest augmenting path from a free row, with the
    standard JV dual update on termination."""
    _, n_col = c.shape
    dist = c[free_row] - u[free_row] - v
    pred = np.full(n_col, free_row)
    done = np.zeros(n_col, dtype=bool)
    row_dist = np.full(len(u), _BIG)
    row_dist[free_row] = 0.0
    sink = -1
    d_star = 0.0
    while True:
        j = -1
        d_min = _BIG
        for col in range(n_col):
            if not done[col] and dist[col] < d_min:
                d_min = float(dist[col])
                j = col
        if j == -1:
            raise ValueError("no augmenting path")
        done[j] = True
        i_star = int(col2row[j])
        if i_star == -1:
            sink, d_star = j, d_min
            break
        row_dist[i_star] = d_min
        rc = c[i_star] - u[i_star] - v
        nxt = d_min + rc
        better = (~done) & (nxt < dist)
        dist[better] = nxt[better]
        pred[better] = i_star
    # dual updates: scanned rows gain d*−d_i, marked columns lose d*−d_j
    for i in range(len(u)):
        if row_dist[i] < _BIG:
            u[i] += d_star - row_dist[i]
    for j in range(n_col):
        if done[j]:
            v[j] += dist[j] - d_star
    # augment along predecessors
    jj = sink
    while True:
        ii = int(pred[jj])
        col2row[jj] = ii
        prev = int(row2col[ii])
        row2col[ii] = jj
        jj = prev
        if jj == -1:
            break


def jv_assign(cost: FloatArray, maximize: bool = False) -> tuple[np.ndarray, float]:
    """Optimal row→col assignment; returns (row2col, total_cost).

    Rectangular inputs are padded to square with column-invariant
    dummy rows — a real row never prefers a dummy column."""
    c = np.asarray(cost, dtype=np.float64)
    if maximize:
        c = -c
    n_row, n_col = c.shape
    if n_row > n_col:
        raise ValueError("needs cols >= rows (pad with dummy cols)")
    if n_col > n_row:
        fill = float(c.max()) + 1.0
        pad = np.full((n_col - n_row, n_col), fill)
        c = np.vstack([c, pad])
    n = c.shape[0]
    # column then row reduction → feasible duals
    v = c.min(axis=0)
    u = (c - v).min(axis=1)
    row2col = np.full(n, -1)
    col2row = np.full(n, -1)
    for r in range(n):
        if row2col[r] == -1:
            _augment(c, u, v, row2col, col2row, r)
    row2col = row2col[:n_row]
    total = float(sum(cost[r, row2col[r]] for r in range(n_row)))
    return row2col, total


def bench_jonker_volgenant(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: agrees with scipy on random costs; identity optimum;
    rectangular padding; deterministic."""
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(seed)
    agree = 0
    worst_gap = 0.0
    for _t in range(12):
        n = int(rng.integers(3, 14))
        m = n + int(rng.integers(0, 5))
        C = rng.uniform(0, 10, (n, m))
        ra, rc = linear_sum_assignment(C)
        ref = float(C[ra, rc].sum())
        row2col, tot = jv_assign(C)
        worst_gap = max(worst_gap, abs(tot - ref))
        agree += int(abs(tot - ref) < 1e-9)
    C_id = np.eye(6) * 0.0 + (1 - np.eye(6)) * 9.0
    a, cost = jv_assign(C_id)
    ident = float((a == np.arange(6)).all())
    return {
        "synthetic_jv_agree": float(agree),
        "synthetic_jv_worst_gap": worst_gap,
        "synthetic_jv_identity": ident,
        "synthetic_jv_identity_cost": float(cost),
    }
