"""Synthetic alignment fixture: context-action rewards and preferences (SYNTHETIC).

Contexts x ~ N(0,I_4); 20 actions with embeddings E_a; true reward
r(x,a) = x·E_a / ||x||·||E_a|| (cosine, bounded). Preference pairs
prefer higher true reward with label noise. All alignment methods
(DPO/IPO/KTO/GRPO/PPO-RLHF) train a softmax policy π(a|x) to maximize
true reward while staying near a reference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_ACT = 20
DX = 4

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def action_embeddings(rng: np.random.Generator) -> FloatArray:
    e = rng.normal(0, 1, (N_ACT, DX))
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


def contexts(n: int, rng: np.random.Generator) -> FloatArray:
    x = rng.normal(0, 1, (n, DX))
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def true_reward(x: FloatArray, emb: FloatArray) -> FloatArray:
    return x @ emb.T


def pref_pairs(n: int, emb: FloatArray, rng: np.random.Generator, noise: float = 0.1):
    """(x, winner, loser) triples with label noise."""
    x = contexts(n, rng)
    r = true_reward(x, emb)
    a_w = np.zeros(n, dtype=np.int64)
    a_l = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pair = rng.choice(N_ACT, 2, replace=False)
        w = pair[int(r[i, pair[0]] < r[i, pair[1]])]
        ll = pair[int(r[i, pair[0]] >= r[i, pair[1]])]
        if rng.random() < noise:
            w, ll = ll, w
        a_w[i], a_l[i] = w, ll
    return x, a_w, a_l


def best_action_rate(policy_logits: FloatArray, x: FloatArray, emb: FloatArray) -> float:
    """Fraction of contexts where argmax_π equals the true-best action."""
    pred = policy_logits.argmax(-1)
    best = true_reward(x, emb).argmax(-1)
    return float((pred == best).mean())
