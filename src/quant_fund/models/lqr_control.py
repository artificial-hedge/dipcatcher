"""Finite-horizon LQR (Riccati recursion) on the double-integrator — (SYNTHETIC)
exact optimal controller; cost vs PD baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._oc_synth import DT, HORIZON, pd_baseline


def _lqr_gain(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, T: int
) -> list[np.ndarray]:
    P = Q.copy()
    Ks = []
    for _ in range(T):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
        Ks.append(K)
    return Ks[::-1]


def bench_lqr_control(seed: int = 2909) -> dict[str, float]:
    A = np.array([[1.0, DT], [0.0, 1.0]])
    B = np.array([[0.0], [DT]])
    Q = np.array([[1.0, 0.0], [0.0, 0.01]])
    R = np.array([[0.001]])
    target = 1.0
    # shift coordinates so target is 0
    Ks = _lqr_gain(A, B, Q, R, HORIZON)
    x0 = np.array([0.0, 0.0])
    x = x0 - np.array([target, 0.0])
    tot = 0.0
    for t in range(HORIZON):
        u = float(-(Ks[t] @ x)[0])
        x = A @ x + B[:, 0] * u
        tot += float(x @ Q @ x + R[0, 0] * u**2)
    _, cost_b = pd_baseline(x0, target)
    return {
        "synthetic_lqr_cost": float(tot),
        "synthetic_pd_cost": float(cost_b),
        "synthetic_lqr_gain": float(cost_b - tot),
        "synthetic_torch_available": 0.0,
    }
