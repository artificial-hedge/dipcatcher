"""PBVI (Pineau, Gordon, Thrun 2003) — point-based value iteration:
maintain a belief set B and one α-vector per point; each backup
greedily assigns each belief its maximizing vector. Complexity is
polynomial in |B| rather than exponential in the horizon.
"""

from __future__ import annotations

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


def point_backup(p: POMDP, b: FloatArray, alphas: FloatArray) -> FloatArray:
    """PBVI backup: for each action build the best α for b via
    per-observation maximizing transformed vectors; return the best
    action's vector."""
    best_v = None
    best_val = -np.inf
    for a in range(p.n_a):
        v = p.r[a].copy()
        for o in range(p.n_o):
            trans = np.array([p.o[a, :, o] * (p.t[a].T @ al) for al in alphas])
            # PBVI: pick the transformed vector best at the POSTERIOR
            bo = belief_update(p, b, a, o)
            scores = trans @ bo
            v = v + p.gamma * trans[int(np.argmax(scores))]
        val = float(v @ b)
        if val > best_val:
            best_val, best_v = val, v
    out: FloatArray = np.asarray(best_v)
    return out


def expand_beliefs(
    p: POMDP,
    beliefs: list[FloatArray],
    rng: np.random.Generator,
    n_new: int,
) -> None:
    """Grow the belief set by random action/obs exploration from
    existing points."""
    for _ in range(n_new):
        b = beliefs[int(rng.integers(len(beliefs)))]
        a = int(rng.integers(p.n_a))
        probs = np.array([obs_prob(p, b, a, o) for o in range(p.n_o)])
        probs = probs / probs.sum()
        o = int(rng.choice(p.n_o, p=probs))
        nb = belief_update(p, b, a, o)
        beliefs.append(nb)


def pbvi(
    p: POMDP,
    n_beliefs: int,
    n_iters: int,
    rng: np.random.Generator,
) -> FloatArray:
    """PBVI: expanding belief set + greedy per-point backups."""
    beliefs: list[FloatArray] = [p.s0_dist.copy()]
    alphas = np.array([np.full(p.n_s, p.r.min() / (1 - p.gamma))])
    for i in range(n_iters):
        # backup stage: new vector per belief point
        new_alpha = np.array([point_backup(p, b, alphas) for b in beliefs])
        alphas = new_alpha
        if i < n_iters - 1 and len(beliefs) < n_beliefs:
            expand_beliefs(p, beliefs, rng, min(3, n_beliefs - len(beliefs)))
    out: FloatArray = np.asarray(alphas)
    return out


def pbvi_policy(p: POMDP, b: FloatArray, alphas: FloatArray) -> int:
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


def bench_pbvi(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: PBVI on Tiger — value approaches the true optimum
    and the induced policy listens before opening."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    p = tiger_pomdp()
    alphas = pbvi(p, n_beliefs=40, n_iters=60, rng=rng)
    out["synthetic_pbvi_nvectors"] = float(len(alphas))
    v0 = float(np.max(alphas @ p.s0_dist))
    out["synthetic_pbvi_value"] = v0
    acts = [pbvi_policy(p, np.array([q, 1 - q]), alphas) for q in np.linspace(0.1, 0.9, 9)]
    out["synthetic_pbvi_listens_mid"] = float(0 in acts[3:6])
    ret = pomdp_rollout(p, lambda b: pbvi_policy(p, b, alphas), 150, 15, rng)
    out["synthetic_pbvi_return"] = ret
    out["synthetic_pbvi_positive"] = float(ret > 0)
    return out


if __name__ == "__main__":
    print(bench_pbvi())
