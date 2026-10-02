"""Perseus (Spaan & Vlassis 2005) — randomized point-based VI: each
epoch backs up a random subset of belief points, reusing each new
α-vector wherever it already improves over the current set. Value
converges with far fewer backups than PBVI.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.pbvi import pbvi_policy, point_backup
from quant_fund.models.qmdp import (
    POMDP,
    pomdp_rollout,
    tiger_pomdp,
)

FloatArray = NDArray[np.float64]


def sample_beliefs(
    p: POMDP, n: int, rng: np.random.Generator, horizon: int = 10
) -> list[FloatArray]:
    """Collect reachable beliefs via random-action rollouts."""
    beliefs = [p.s0_dist.copy()]
    from quant_fund.models.qmdp import belief_update, obs_prob

    b = p.s0_dist.copy()
    for _ in range(n):
        a = int(rng.integers(p.n_a))
        probs = np.array([obs_prob(p, b, a, o) for o in range(p.n_o)])
        if probs.sum() <= 0:
            b = p.s0_dist.copy()
            continue
        o = int(rng.choice(p.n_o, p=probs / probs.sum()))
        b = belief_update(p, b, a, o)
        beliefs.append(b)
        if rng.random() < 0.1:
            b = p.s0_dist.copy()
    return beliefs


def perseus(
    p: POMDP,
    n_beliefs: int,
    n_epochs: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Perseus VI: random-subset backups until all points improved."""
    beliefs = sample_beliefs(p, n_beliefs, rng)
    bmat = np.array(beliefs)
    alphas = np.array([np.full(p.n_s, p.r.min() / (1 - p.gamma))])
    for _ in range(n_epochs):
        unimproved = set(range(len(beliefs)))
        vals = np.max(bmat @ alphas.T, axis=1)
        order = rng.permutation(len(beliefs))
        new_alphas: list[FloatArray] = []
        for i in order:
            if i not in unimproved:
                continue
            a_vec = point_backup(p, bmat[i], alphas)
            # if the new vector doesn't improve its own point, keep max
            if float(a_vec @ bmat[i]) < vals[i]:
                # reuse the incumbent maximizer at b_i
                a_vec = alphas[int(np.argmax(alphas @ bmat[i]))]
            new_alphas.append(a_vec)
            # mark every point this vector improves as improved
            for j in list(unimproved):
                if float(a_vec @ bmat[j]) >= vals[j] - 1e-12:
                    unimproved.discard(j)
            if not unimproved:
                break
        if new_alphas:
            alphas = np.array(new_alphas)
    out: FloatArray = np.asarray(alphas)
    return out


def bench_perseus(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: Perseus on Tiger — small α-set, competitive value
    and return with the point-based policy listening mid-belief."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    p = tiger_pomdp()
    alphas = perseus(p, n_beliefs=60, n_epochs=80, rng=rng)
    out["synthetic_perseus_nvectors"] = float(len(alphas))
    v0 = float(np.max(alphas @ p.s0_dist))
    out["synthetic_perseus_value"] = v0
    acts = [pbvi_policy(p, np.array([q, 1 - q]), alphas) for q in np.linspace(0.1, 0.9, 9)]
    out["synthetic_perseus_listens_mid"] = float(0 in acts[3:6])
    ret = pomdp_rollout(p, lambda b: pbvi_policy(p, b, alphas), 150, 15, rng)
    out["synthetic_perseus_return"] = ret
    out["synthetic_perseus_positive"] = float(ret > 0)
    return out


if __name__ == "__main__":
    print(bench_perseus())
