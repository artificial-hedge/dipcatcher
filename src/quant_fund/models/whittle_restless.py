"""Whittle index for restless bandits — two-state Markov arms
(active/passive transition matrices); Whittle threshold per state;
play top-m arms by index each step vs round-robin.
"""

from __future__ import annotations

import numpy as np


def _whittle(Pa: np.ndarray, Pp: np.ndarray, R: np.ndarray, grid: int = 41) -> np.ndarray:
    # approximate Whittle: for each state, threshold subsidy m where
    # active vs passive action values tie under discounted VI
    out = np.zeros(2)
    ms = np.linspace(0, 1, grid)
    for s in range(2):
        prev_diff = None
        for m in ms:
            V = np.zeros(2)
            Ra = R.copy()
            Ra[s] = max(R[s], m) if s == s else R[s]
            for _ in range(60):
                Va = R + 0.95 * Pa @ V
                Vp = np.minimum.accumulate(np.array([m, m])) + 0.95 * Pp @ V
                V = np.maximum(Va, Vp)
            diff = (R[s] + 0.95 * Pa[s] @ V) - (m + 0.95 * Pp[s] @ V)
            if prev_diff is not None and prev_diff < 0 <= diff:
                out[s] = m
                break
            prev_diff = diff
    return out


def bench_whittle_restless(
    seed: int = 1411, K: int = 6, T: int = 3000, m_play: int = 3
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    Pa = np.stack([rng.dirichlet([3, 1]) for _ in range(K)])
    Pp = np.stack([rng.dirichlet([1, 3]) for _ in range(K)])
    R = rng.uniform(0.2, 1.0, (K, 2))
    state = rng.integers(0, 2, K)
    W = np.stack(
        [_whittle(Pa[k : k + 1].repeat(2, 0), Pp[k : k + 1].repeat(2, 0), R[k]) for k in range(K)]
    )
    tot = 0.0
    for _ in range(T):
        idx = W[range(K), state]
        play = np.argsort(-idx)[:m_play]
        for k in play:
            tot += R[k, state[k]]
            state[k] = int(rng.choice(2, p=Pa[k]))
        for k in range(K):
            if k not in play:
                state[k] = int(rng.choice(2, p=Pp[k]))
    # round-robin
    state2 = rng.integers(0, 2, K)
    tot2 = 0.0
    for t in range(T):
        play = np.array([(t * m_play + i) % K for i in range(m_play)])
        for k in play:
            tot2 += R[k, state2[k]]
            state2[k] = int(rng.choice(2, p=Pa[k]))
        for k in range(K):
            if k not in play:
                state2[k] = int(rng.choice(2, p=Pp[k]))
    return {
        "synthetic_whi_mean_reward": tot / T,
        "synthetic_whi_rr_reward": tot2 / T,
        "synthetic_whi_reward_gain": (tot - tot2) / T,
        "torch_available": 0.0,
    }
