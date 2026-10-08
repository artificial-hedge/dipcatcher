"""LQG = Kalman filter + LQR separation on a noisy double integrator: (SYNTHETIC)
observations carry position noise; controller uses the filtered
estimate. Cost vs LQR driven by raw measurements.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._oc_synth import DT, HORIZON
from quant_fund.models.lqr_control import _lqr_gain


def bench_lqg_control(
    seed: int = 2929, meas_sd: float = 0.4, proc_sd: float = 0.15
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A = np.array([[1.0, DT], [0.0, 1.0]])
    B = np.array([[0.0], [DT]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1.0, 0.0], [0.0, 0.01]])
    R = np.array([[0.001]])
    W = proc_sd**2 * np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])
    V = np.array([[meas_sd**2]])
    target = 1.0
    Ks = _lqr_gain(A, B, Q, R, HORIZON)

    x_true = np.array([0.0, 0.0]) - np.array([target, 0.0])
    cost_f, cost_r = 0.0, 0.0
    # filtered controller
    xh = x_true.copy()
    P = np.eye(2) * 0.5
    x_true2 = x_true.copy()
    x_meas = x_true.copy()
    for t in range(HORIZON):
        # truth evolves with process noise
        w = rng.multivariate_normal(np.zeros(2), W)
        # LQG arm
        z = (x_true + np.array([target, 0]))[0] + rng.normal(0, meas_sd)
        u_f = float(-(Ks[t] @ xh)[0])
        x_true = A @ x_true + B[:, 0] * u_f + w
        cost_f += float(x_true @ Q @ x_true + R[0, 0] * u_f**2)
        # KF update (in shifted coords: observe z-target)
        y = np.array([z - target])
        xh = A @ xh + B[:, 0] * u_f
        P = A @ P @ A.T + W
        S = H @ P @ H.T + V
        Kg = P @ H.T @ np.linalg.inv(S)
        xh = xh + (Kg @ (y - H @ xh))
        P = (np.eye(2) - Kg @ H) @ P
        # raw-measurement arm (same disturbance realizations approximated)
        u_r = float(-(Ks[t] @ x_meas)[0])
        x_true2 = A @ x_true2 + B[:, 0] * u_r + w
        cost_r += float(x_true2 @ Q @ x_true2 + R[0, 0] * u_r**2)
        x_meas = x_true2.copy()
        x_meas[0] = z - target
    return {
        "synthetic_lqg_cost": float(cost_f),
        "synthetic_raw_lqr_cost": float(cost_r),
        "synthetic_lqg_gain": float(cost_r - cost_f),
        "synthetic_torch_available": 0.0,
    }
