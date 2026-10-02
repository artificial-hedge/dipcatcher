"""Value/policy-iteration contraction vs the Bellman gamma bound.

Two checks on a planted 4-state, 2-action MDP: (a) value iteration
satisfies ||v_{k+1} - v*|| <= gamma ||v_k - v*|| at every sweep;
(b) exact policy iteration reaches the same fixed point in few steps.
"""

import numpy as np

from quant_fund.models._rlt_synth import MDP_GAMMA

# 4 states; action 0 = go left, 1 = go right (deterministic, clamped)
S_R = np.array([0.0, 0.2, 0.4, 2.0])


def _trans(s: int, a: int) -> int:
    return max(0, s - 1) if a == 0 else min(len(S_R) - 1, s + 1)


def _vi(iters: int = 500) -> np.ndarray:
    v = np.zeros(len(S_R))
    for _ in range(iters):
        v = np.array(
            [S_R[s] + MDP_GAMMA * max(v[_trans(s, 0)], v[_trans(s, 1)]) for s in range(len(S_R))]
        )
    return v


def _vi_errs() -> list[float]:
    v = np.zeros(len(S_R))
    v_star = _vi()
    errs = [float(np.max(np.abs(v - v_star)))]
    for _ in range(80):
        v = np.array(
            [S_R[s] + MDP_GAMMA * max(v[_trans(s, 0)], v[_trans(s, 1)]) for s in range(len(S_R))]
        )
        errs.append(float(np.max(np.abs(v - v_star))))
        if errs[-1] < 1e-12:
            break
    return errs


def _pi() -> tuple[int, np.ndarray]:
    n = len(S_R)
    pol = np.zeros(n, dtype=int)
    for it in range(1, 30):
        p = np.array([[1.0 if _trans(s, pol[s]) == t else 0.0 for t in range(n)] for s in range(n)])
        v = np.linalg.solve(np.eye(n) - MDP_GAMMA * p, S_R)
        new = np.array([int(v[_trans(s, 1)] >= v[_trans(s, 0)]) for s in range(n)], dtype=int)
        if np.array_equal(new, pol):
            return it, pol
        pol = new
    return 30, pol


def bench_pi_contraction(seed: int = 4607) -> dict[str, float]:
    del seed
    errs = _vi_errs()
    ratios = [errs[i + 1] / errs[i] for i in range(len(errs) - 1) if errs[i] > 1e-9]
    pi_iters, pol = _pi()
    n = len(S_R)
    p_pi = np.array([[1.0 if _trans(s, pol[s]) == t else 0.0 for t in range(n)] for s in range(n)])
    v_pi = np.linalg.solve(np.eye(n) - MDP_GAMMA * p_pi, S_R)
    return {
        "synthetic_pi_max_ratio": float(max(ratios)),
        "synthetic_pi_gamma": MDP_GAMMA,
        "synthetic_pi_vi_sweeps": float(len(errs) - 1),
        "synthetic_pi_iters": float(pi_iters),
        "synthetic_pi_v_err": float(np.max(np.abs(v_pi - _vi()))),
    }
