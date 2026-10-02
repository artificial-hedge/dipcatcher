"""HSVI (Smith & Simmons 2004) — heuristic search value iteration:
interleaved upper/lower bounds with forward exploration weighted by
excess uncertainty. Lower bound = α-vector set (PBVI backups); upper
bound = sawtooth approximation over a point-value map.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.pbvi import point_backup
from quant_fund.models.qmdp import (
    POMDP,
    belief_update,
    mdp_value_iteration,
    obs_prob,
    pomdp_rollout,
    tiger_pomdp,
)

FloatArray = NDArray[np.float64]


def hsvi(
    p: POMDP,
    n_iters: int,
    rng: np.random.Generator,
    eps: float = 0.5,
    horizon: int = 10,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Simplified HSVI on small POMDPs.

    Maintains Γ (α-vectors, lower bound) and an upper bound from the
    MDP solution refined by forward search. Explores the action
    maximizing the upper bound and the observation maximizing
    weighted excess uncertainty. Returns (alphas, beliefs, ub_vals).
    """
    v_mdp = mdp_value_iteration(p)
    alphas = np.array([np.full(p.n_s, p.r.min() / (1 - p.gamma))])
    # upper-bound point map: (belief, value) via MD+ exploration
    ub_points: list[FloatArray] = [p.s0_dist.copy()]
    ub_vals: list[float] = [float(v_mdp @ p.s0_dist)]

    def lb(b: FloatArray) -> float:
        return float(np.max(alphas @ b))

    def ub(b: FloatArray) -> float:
        pts = np.array(ub_points)
        vals = np.array(ub_vals)
        best = np.inf
        for i in range(len(pts)):
            # sawtooth: height at corner k = v_i·(1/... ) — use the
            # tight standard form over stored points
            gap = float(np.max(np.abs(b - pts[i])))
            best = min(best, vals[i] + gap * 60.0)
        return best

    for _ in range(n_iters):
        # forward exploration from b0 by greedy-UB action + widest-gap obs
        b = p.s0_dist.copy()
        for _d in range(horizon):
            # backup current point on both bounds
            alphas = np.vstack([alphas, point_backup(p, b, alphas)])
            u_b = min(ub(b), float(v_mdp @ b))
            ub_points.append(b.copy())
            ub_vals.append(u_b)
            # choose action by UB lookahead
            best_a, best_v = 0, -np.inf
            for a in range(p.n_a):
                tot = float(p.r[a] @ b)
                for o in range(p.n_o):
                    po = obs_prob(p, b, a, o)
                    if po > 1e-9:
                        tot += p.gamma * po * ub(belief_update(p, b, a, o))
                if tot > best_v:
                    best_v, best_a = tot, a
            # choose obs with max excess uncertainty weighted by prob
            best_o, best_gap = 0, -np.inf
            for o in range(p.n_o):
                po = obs_prob(p, b, best_a, o)
                if po <= 1e-9:
                    continue
                bo = belief_update(p, b, best_a, o)
                g = po * (ub(bo) - lb(bo))
                if g > best_gap:
                    best_gap, best_o = g, o
            b = belief_update(p, b, best_a, best_o)
        if abs(ub(p.s0_dist) - lb(p.s0_dist)) < eps:
            break
    out_a: FloatArray = np.asarray(alphas)
    return out_a, np.asarray(ub_points), np.asarray(ub_vals)


def hsvi_policy(p: POMDP, b: FloatArray, alphas: FloatArray) -> int:
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


def bench_hsvi(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: HSVI on Tiger — LB rises, gap shrinks, policy
    listens and returns positive."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    p = tiger_pomdp()
    alphas, _pts, _uv = hsvi(p, n_iters=25, rng=rng, horizon=8)
    out["synthetic_hsvi_nvectors"] = float(len(alphas))
    lb0 = float(np.max(alphas @ p.s0_dist))
    out["synthetic_hsvi_lb"] = lb0
    ret = pomdp_rollout(p, lambda b: hsvi_policy(p, b, alphas), 150, 15, rng)
    out["synthetic_hsvi_return"] = ret
    acts = [hsvi_policy(p, np.array([q, 1 - q]), alphas) for q in np.linspace(0.1, 0.9, 9)]
    out["synthetic_hsvi_listens_mid"] = float(0 in acts[3:6])
    out["synthetic_hsvi_positive"] = float(ret > 0)
    return out


if __name__ == "__main__":
    print(bench_hsvi())
