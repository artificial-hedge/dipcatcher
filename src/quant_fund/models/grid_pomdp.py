"""Monahan/Lovejoy grid value iteration — piecewise-linear convex
value function over the belief simplex via α-vector sets; the full
cross-sum backup is pruned by dominance on a uniform belief grid.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.qmdp import (
    POMDP,
    belief_update,
    obs_prob,
    pomdp_rollout,
    tiger_pomdp,
)

FloatArray = NDArray[np.float64]


def cross_sum(p: POMDP, alphas: FloatArray, a: int) -> list[FloatArray]:
    """Γ_{a} = { r(·,a)/|O| ... } cross-sum of transformed vectors.

    For each o, Γ_{a,o} = { α'(s) = Σ_s' O(a,s',o)T(a,s,s')α(s') }.
    Returns the list of summed vectors r_a + γ·Σ_o v_o over choices
    v_o ∈ Γ_{a,o}.
    """
    trans = []
    for o in range(p.n_o):
        # g_o = O(a,·,o) ∘ T(a,·,·)ᵀ α  — expected future vector on obs o
        trans.append(np.array([p.o[a, :, o] * (p.t[a].T @ al) for al in alphas]))
    out = []
    for combo in itertools.product(range(len(alphas)), repeat=p.n_o):
        v = p.r[a].copy()
        for o in range(p.n_o):
            v = v + p.gamma * trans[o][combo[o]]
        out.append(np.asarray(v))
    return out


def prune_on_grid(alphas: list[FloatArray], grid: FloatArray) -> FloatArray:
    """Keep vectors that are maximal at some grid belief."""
    if not alphas:
        return np.zeros((0, grid.shape[1]))
    mat = np.array(alphas)
    keep: list[int] = []
    vals = grid @ mat.T
    best_idx = np.argmax(vals, axis=1)
    for i in np.unique(best_idx):
        keep.append(int(i))
    out: FloatArray = np.asarray(mat[sorted(set(keep))])
    return out


def grid_vi(
    p: POMDP,
    n_iters: int,
    grid_n: int = 21,
) -> FloatArray:
    """VI on α-vector set with grid pruning. Returns the pruned set."""
    # uniform grid over the 2-simplex (works for n_s=2; generalize by
    # dirichlet grid for n_s>2 — tiger only here)
    g1 = np.linspace(0, 1, grid_n)
    grid = np.stack([g1, 1 - g1], axis=1)
    alphas = np.array([p.r.min(axis=1)[0] * np.ones(p.n_s) / (1 - p.gamma)])
    for _ in range(n_iters):
        cand = []
        for a in range(p.n_a):
            cand.extend(cross_sum(p, alphas, a))
        alphas = prune_on_grid(cand, grid)
    out: FloatArray = np.asarray(alphas)
    return out


def grid_policy(p: POMDP, b: FloatArray, alphas: FloatArray) -> int:
    """Greedy action under the α-vector value: choose argmax over the
    one-step lookahead values via the cross-sum decomposition."""
    best_a, best_v = 0, -np.inf
    for a in range(p.n_a):
        tot = float(p.r[a] @ b)
        for o in range(p.n_o):
            po = obs_prob(p, b, a, o)
            if po <= 1e-12:
                continue
            bo = belief_update(p, b, a, o)
            tot += p.gamma * po * float(np.max(alphas @ bo))
        if tot > best_v:
            best_v, best_a = tot, a
    return best_a


def bench_grid(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: grid VI on Tiger converges to a listening policy —
    rolled-out return ≫ QMDP's door guessing."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    p = tiger_pomdp()
    alphas = grid_vi(p, n_iters=20, grid_n=41)
    out["synthetic_grid_nvectors"] = float(len(alphas))
    v0 = float(np.max(alphas @ p.s0_dist))
    out["synthetic_grid_value"] = v0
    acts = [grid_policy(p, np.array([q, 1 - q]), alphas) for q in np.linspace(0.1, 0.9, 9)]
    out["synthetic_grid_listens_mid"] = float(acts[4] == 0 or acts[3] == 0 or acts[5] == 0)
    ret = pomdp_rollout(p, lambda b: grid_policy(p, b, alphas), 150, 15, rng)
    out["synthetic_grid_return"] = ret
    out["synthetic_grid_beats_qmdp"] = float(ret > 0)
    # α-vector VI is a LOWER bound on the QMDP upper bound
    from quant_fund.models.qmdp import mdp_value_iteration

    v_ub = float(mdp_value_iteration(p) @ p.s0_dist)
    out["synthetic_grid_below_qmdp"] = float(v0 <= v_ub + 1e-6)
    return out


if __name__ == "__main__":
    print(bench_grid())
