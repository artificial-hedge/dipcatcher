"""Zero-sum stochastic game via Shapley value iteration (SYNTHETIC).

Each stage game solves the 2x2 minimax of M + gamma * E[v(next)|actions]
by the closed-form 2x2 value (saddle check, then mixed-strategy formula).
Bench: Bellman residual after convergence and the equilibrium value's gap
over the pure worst-case floor.
"""

import numpy as np

from quant_fund.models._mfg_synth import SG_GAMMA, SG_M, SG_P


def _matrix_value(m: np.ndarray) -> float:
    mins = float(m.max(axis=1).min())
    maxs = float(m.min(axis=0).max())
    if mins <= maxs + 1e-12:
        return maxs
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    denom = a - b - c + d
    return float((a * d - b * c) / denom)


def _cont(s: int, v: np.ndarray) -> np.ndarray:
    return np.array([[SG_P[s][i, j] @ v for j in range(2)] for i in range(2)])


def _vi() -> np.ndarray:
    v = np.zeros(2)
    for _ in range(500):
        nv = np.array([_matrix_value(SG_M[s] + SG_GAMMA * _cont(s, v)) for s in range(2)])
        if np.max(np.abs(nv - v)) < 1e-12:
            return nv
        v = nv
    return v


def bench_stochastic_game_vi(seed: int = 4209) -> dict[str, float]:
    del seed
    v = _vi()
    resid = max(abs(_matrix_value(SG_M[s] + SG_GAMMA * _cont(s, v)) - v[s]) for s in range(2))
    naive = float(min(float(SG_M[s].min()) for s in range(2)))
    return {
        "synthetic_sg_resid": float(resid),
        "synthetic_sg_v0": float(v[0]),
        "synthetic_sg_v1": float(v[1]),
        "synthetic_sg_naive_floor": naive,
        "synthetic_sg_floor_gap": float(v[0] - naive),
    }
