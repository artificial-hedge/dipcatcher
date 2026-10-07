"""Linear MPC via batch-QP: condense the finite-horizon LQR into a QP
in u_0..u_{T-1}, solve by projected gradient. Receding-horizon cost
vs PD baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._oc_synth import DT, pd_baseline


def bench_mpc_qp(seed: int = 2925, horizon: int = 20, steps: int = 30) -> dict[str, float]:
    A = np.array([[1.0, DT], [0.0, 1.0]])
    B = np.array([[0.0], [DT]])
    Q = np.array([[1.0, 0.0], [0.0, 0.01]])
    R = 0.001
    target = 1.0
    T = horizon

    # condense: x_t = A^t x0 + sum_{k<t} A^{t-1-k} B u_k
    Sx = np.zeros((T * 2, 2))
    Su = np.zeros((T * 2, T))
    for t in range(T):
        Sx[2 * t : 2 * t + 2] = np.linalg.matrix_power(A, t + 1)
        for k in range(t + 1):
            Su[2 * t : 2 * t + 2, k] = np.linalg.matrix_power(A, t - k) @ B[:, 0]
    Qbar = np.kron(np.eye(T), Q)
    H = Su.T @ Qbar @ Su + R * np.eye(T)

    x = np.array([0.0, 0.0]) - np.array([target, 0.0])
    tot = 0.0
    u_hist: list[float] = []
    for _ in range(steps):
        g = Su.T @ Qbar @ (Sx @ x)
        # projected gradient on |u|<=2
        u = np.zeros(T)
        L = float(np.linalg.eigvalsh(H).max())
        for _ in range(200):
            u = np.clip(u - (H @ u + g) / L, -2, 2)
        a = float(np.clip(u[0], -2, 2))
        u_hist.append(a)
        x = A @ x + B[:, 0] * a
        tot += float(x @ Q @ x + R * a * a)
    _, cost_b = pd_baseline(np.array([0.0, 0.0]), target)
    # PD cost over same horizon
    xs_b, _ = pd_baseline(np.array([0.0, 0.0]), target)
    # recompute PD over `steps` steps
    xpd = np.array([0.0, 0.0]) - np.array([target, 0.0])
    cost_pd = 0.0
    for _ in range(steps):
        a = np.clip(0.6 * (0 - xpd[0]) - 1.2 * xpd[1], -2, 2)
        xpd = A @ xpd + B[:, 0] * a
        cost_pd += float(xpd @ Q @ xpd + R * a * a)
    return {
        "synthetic_mpc_cost": float(tot),
        "synthetic_pd_cost": float(cost_pd),
        "synthetic_mpc_gain": float(cost_pd - tot),
        "synthetic_torch_available": 0.0,
    }
