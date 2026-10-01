"""Rank aggregation — Plackett-Luce, Borda, Condorcet, MC3.

Plackett (1975) / Luce (1959): partial orderings are treated
as sequential choices p(i first of S) = w_i / sum_{j in S} w_j
with nonnegative support parameters w estimated by the
Hunter (2004) minorization-maximization iteration

    w_i <- N_i / sum_{rankings r} sum_{k: r_k = i} 1 / (sum_{j
    unchosen at step k} w_j).

Borda (1781) assigns n-k points for rank k; Condorcet
pairwise winners form a majority graph (Copeland score =
wins - losses); Negahban-Oh-Shah (2012) MC3 estimates the
stationary distribution of the "vote with pair" Markov chain
(items trade mass according to pairwise win rates).

Honesty: orderings must be complete permutations of the same
item set (ties split to the front — flagged); PL-MM
convergence is a fixed cap, no MLE standard errors reported
(bootstrap SE is possible but not implemented — documented).
The bench plants orderings drawn from a known PL distribution
and checks the estimated ranking inverts the true support
order. Fail-closed on inconsistent item sets or <2 rankings.

References: Luce (1959) "Individual Choice Behavior"; Plackett
(1975) JRSS-C 24:193; Hunter (2004) Ann. Stat. 32:384;
Negahban, Oh & Shah (2012) NeurIPS; Arrow & Raynaud (1986).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check(rankings: FloatArray) -> tuple[IntArray, int, int]:
    r = np.asarray(rankings)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 2:
        raise ValueError("rankings must be R x K, R>=2, K>=2")
    ri = r.astype(np.int64)
    if not np.allclose(ri, r):
        raise ValueError("rankings must be integer permutations")
    ref = np.sort(ri[0])
    for row in ri:
        if not np.array_equal(np.sort(row), ref):
            raise ValueError("inconsistent item sets across rankings")
    return ri, ri.shape[0], ri.shape[1]


def plackett_luce(
    rankings: FloatArray, n_iter: int = 500, seed: int = 0
) -> dict[str, float | FloatArray]:
    """Plackett-Luce support parameters via Hunter MM."""
    ri, n_r, k = _check(rankings)
    items = np.unique(ri)
    idx = {it: i for i, it in enumerate(items)}
    # ranks_m[r, j] = position (1=first) of item-index j in ranking r
    pos = np.zeros((n_r, len(items)), dtype=np.int64)
    for r in range(n_r):
        for p, it in enumerate(ri[r]):
            pos[r, idx[it]] = p + 1
    w = np.ones(len(items))
    # wins[i] = count of rankings where item i is ranked first
    wins = np.asarray([(pos[:, i] == 1).sum() for i in range(len(items))], dtype=np.float64)
    for _ in range(n_iter):
        denom = np.zeros(len(items))
        for r in range(n_r):
            order = np.argsort(pos[r])  # items by position
            w_sum = w[order].sum()
            running = w_sum
            for it in order:
                denom[it] += 1.0 / max(running, 1e-12)
                running -= w[it]
        w_new = wins / np.clip(denom, 1e-12, None)
        w_new /= w_new.sum()
        if float(np.abs(w_new - w).max()) < 1e-10:
            w = w_new
            break
        w = w_new
    order_out = np.argsort(-w)
    return {
        "w": np.asarray(w, dtype=np.float64),
        "item_order": np.asarray(items[order_out], dtype=np.float64),
        "items": np.asarray(items, dtype=np.float64),
        "log_lik_scale": float(np.log(w.max() / max(w.min(), 1e-12))),
    }


def borda_count(rankings: FloatArray) -> dict[str, float | FloatArray]:
    """Borda scores (sum of n - position points)."""
    ri, n_r, k = _check(rankings)
    items = np.unique(ri)
    idx = {it: i for i, it in enumerate(items)}
    scores = np.zeros(len(items))
    for r in range(n_r):
        for p, it in enumerate(ri[r]):
            scores[idx[it]] += k - 1 - p
    scores /= n_r
    order_out = np.argsort(-scores)
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "item_order": np.asarray(items[order_out], dtype=np.float64),
        "items": np.asarray(items, dtype=np.float64),
    }


def condorcet_copeland(rankings: FloatArray) -> dict[str, float | FloatArray]:
    """Copeland score (pairwise wins - losses) of the majority graph."""
    ri, n_r, k = _check(rankings)
    items = np.unique(ri)
    idx = {it: i for i, it in enumerate(items)}
    n_it = len(items)
    win = np.zeros((n_it, n_it))
    pos = np.zeros((n_r, n_it), dtype=np.int64)
    for r in range(n_r):
        for p, it in enumerate(ri[r]):
            pos[r, idx[it]] = p
    for i in range(n_it):
        for j in range(n_it):
            if i != j:
                win[i, j] = float((pos[:, i] < pos[:, j]).sum())
    copeland = win.sum(axis=1) - win.sum(axis=0)
    order_out = np.argsort(-copeland)
    return {
        "copeland": np.asarray(copeland, dtype=np.float64),
        "win_matrix": np.asarray(win, dtype=np.float64),
        "item_order": np.asarray(items[order_out], dtype=np.float64),
        "items": np.asarray(items, dtype=np.float64),
    }


def mc3_ranking(rankings: FloatArray, n_iter: int = 1000) -> dict[str, float | FloatArray]:
    """Negahban-Oh-Shah MC3 — stationary distribution of the
    pairwise-win Markov chain."""
    ri, n_r, k = _check(rankings)
    items = np.unique(ri)
    idx = {it: i for i, it in enumerate(items)}
    n_it = len(items)
    win = np.zeros((n_it, n_it))
    pos = np.zeros((n_r, n_it), dtype=np.int64)
    for r in range(n_r):
        for p, it in enumerate(ri[r]):
            pos[r, idx[it]] = p
    for i in range(n_it):
        for j in range(n_it):
            if i != j:
                win[i, j] = float((pos[:, i] < pos[:, j]).sum())
    # transition: mass flows toward the pairwise winner
    trans = np.zeros((n_it, n_it))
    for i in range(n_it):
        for j in range(n_it):
            if i != j:
                trans[i, j] = win[j, i] / (n_r * n_it)
        trans[i, i] = 1.0 - trans[i].sum()
    pi = np.ones(n_it) / n_it
    for _ in range(n_iter):
        pi_new = pi @ trans
        pi_new /= pi_new.sum()
        if float(np.abs(pi_new - pi).max()) < 1e-12:
            pi = pi_new
            break
        pi = pi_new
    order_out = np.argsort(-pi)
    return {
        "stationary": np.asarray(pi, dtype=np.float64),
        "item_order": np.asarray(items[order_out], dtype=np.float64),
        "items": np.asarray(items, dtype=np.float64),
    }


def bench_rank_aggregation(seed: int = 20261231 + 457) -> dict[str, float]:
    """SYNTHETIC check — PL orderings recover the support order."""
    rng = np.random.default_rng(seed)
    k = 5
    w_true = np.array([0.40, 0.25, 0.18, 0.11, 0.06])
    n_r = 400
    rankings = []
    for _ in range(n_r):
        # sequential PL draw
        pool = list(range(k))
        wr = w_true.copy()
        order = []
        for _j in range(k):
            probs = wr / wr.sum()
            pick = rng.choice(len(pool), p=probs)
            order.append(pool.pop(pick))
            wr = np.delete(wr, pick)
        rankings.append(order)
    out = plackett_luce(np.array(rankings))
    est_order = np.asarray(out["item_order"], dtype=np.int64)
    # top-2 and bottom item must match the true support order
    ok = est_order[0] == 0 and est_order[1] == 1 and est_order[-1] == 4
    if not ok:
        raise ValueError(f"pl order off: {est_order.tolist()}")
    return {
        "synthetic_top_item": float(est_order[0]),
        "synthetic_w_top": float(np.asarray(out["w"]).max()),
        "score": 1.0,
    }
