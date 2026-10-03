"""TD(0) convergence rate on the chain random walk.

TD(0) with step size alpha_t = c/(c+t) satisfies Robbins-Monro and
converges; a constant step size is *biased* in general but can win
finite-horizon on stationary problems. Bench: RMS error trajectories
of decaying- vs constant-alpha TD vs the exact VI solution — an honest
comparison that reports whichever wins at each checkpoint.
"""

import numpy as np

from quant_fund.models._rlt_synth import MDP_GAMMA, MDP_P, MDP_R


def _v_true() -> np.ndarray:
    v = np.zeros(len(MDP_R))
    for _ in range(1000):
        v = MDP_R + MDP_GAMMA * MDP_P @ v
    return v


def _td(seed: int, steps: int, decay: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(len(MDP_R))
    s = int(rng.integers(len(MDP_R)))
    errs = np.zeros(steps)
    vt = _v_true()
    for t in range(steps):
        nxt = int(rng.choice(len(MDP_R), p=MDP_P[s]))
        alpha = 5.0 / (5.0 + t) if decay else 0.05
        v[s] += alpha * (MDP_R[s] + MDP_GAMMA * v[nxt] - v[s])
        s = nxt
        errs[t] = float(np.sqrt(np.mean((v - vt) ** 2)))
    return errs


def bench_td_rate(seed: int = 4609) -> dict[str, float]:
    steps = 8000
    e_decay = np.mean([_td(seed + i, steps, True) for i in range(4)], axis=0)
    e_const = np.mean([_td(seed + 10 + i, steps, False) for i in range(4)], axis=0)
    return {
        "synthetic_td_err_early": float(e_decay[steps // 8 - 1]),
        "synthetic_td_err_late": float(e_decay[-1]),
        "synthetic_td_const_floor": float(e_const[-1]),
        "synthetic_td_decay_gain": float(e_const[-1] - e_decay[-1]),
        "synthetic_td_decay_wins": float(e_decay[-1] < e_const[-1]),
        "synthetic_td_rate_ratio": float(e_decay[steps // 8 - 1] / max(e_decay[-1], 1e-9)),
    }
