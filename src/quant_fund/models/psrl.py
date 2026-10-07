"""PSRL (Osband et al. 2013) — posterior-sampling RL: sample a full MDP
from Dirichlet/multinomial posteriors each epoch, act with its optimal
policy. Cumulative regret vs epsilon-greedy Q-learning.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._bx_synth import psrl_env


def _vi(P: np.ndarray, R: np.ndarray, gamma: float = 0.95) -> np.ndarray:
    V = np.zeros(4)
    for _ in range(200):
        V = np.maximum.reduce([R[a] + gamma * P[a] @ V for a in range(2)])
    return np.argmax([R[a] + gamma * P[a] @ V for a in range(2)], 0)


def bench_psrl(seed: int = 1401, T: int = 3000, epoch: int = 60) -> dict[str, float]:
    P0, P1, R0, R1 = psrl_env(seed)
    rng = np.random.default_rng(seed)
    P_true = np.stack([P0, P1])
    R_true = np.stack([R0, R1])
    # PSRL
    cnt = np.ones((2, 4, 4))
    rcnt = np.ones((2, 4))
    rsum = np.zeros((2, 4))
    s, tot = 0, 0.0
    pol = _vi(P_true, R_true)  # warm init arbitrary
    for t in range(T):
        if t % epoch == 0:
            Ps = np.stack(
                [np.apply_along_axis(lambda v: rng.dirichlet(v), -1, cnt[a]) for a in range(2)]
            )
            Rs = rsum / rcnt + 0.05 * rng.standard_normal((2, 4))
            pol = _vi(Ps, Rs)
        a = pol[s]
        s2 = int(rng.choice(4, p=P_true[a, s]))
        r = R_true[a, s] + 0.05 * rng.standard_normal()
        cnt[a, s, s2] += 1
        rcnt[a, s] += 1
        rsum[a, s] += r
        tot += r
        s = s2
    # epsilon-greedy Q baseline
    Q = np.zeros((4, 2))
    tot2, s = 0.0, 0
    for _t in range(T):
        a = int(rng.integers(2)) if rng.random() < 0.3 else int(np.argmax(Q[s]))
        s2 = int(rng.choice(4, p=P_true[a, s]))
        r = R_true[a, s] + 0.05 * rng.standard_normal()
        Q[s, a] += 0.1 * (r + 0.95 * Q[s2].max() - Q[s, a])
        tot2 += r
        s = s2
    return {
        "synthetic_psrl_total_reward": tot / T,
        "synthetic_psrl_eps_reward": tot2 / T,
        "synthetic_psrl_reward_gain": (tot - tot2) / T,
        "synthetic_torch_available": 0.0,
    }
