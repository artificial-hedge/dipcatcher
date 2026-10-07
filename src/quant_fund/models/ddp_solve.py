"""DDP/iLQR-style shooting optimizer on a nonlinear plant (SYNTHETIC)
(x' = x + dt(v + 0.2 v^2), v' = v + dt·tanh(u)): Gauss-Newton on the
nonlinear dynamics with Riccati backward pass. Cost vs PD baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._oc_synth import DT, HORIZON, pd_baseline


def _dyn(x: np.ndarray, u: float) -> np.ndarray:
    return np.array([x[0] + DT * (x[1] + 0.2 * x[1] ** 2), x[1] + DT * np.tanh(u)])


def _cost(x: np.ndarray, u: float) -> float:
    return float((x[0] - 1.0) ** 2 + 0.01 * x[1] ** 2 + 0.001 * u**2)


def _rollout(u: np.ndarray, x0: np.ndarray) -> tuple[list[np.ndarray], float]:
    x = x0.copy()
    xs = [x.copy()]
    tot = 0.0
    for t in u:
        x = _dyn(x, t)
        tot += _cost(x, t)
        xs.append(x.copy())
    return xs, tot


def bench_ddp_solve(seed: int = 2911, iters: int = 30) -> dict[str, float]:
    x0 = np.array([0.0, 0.0])
    u = np.zeros(HORIZON)
    for _ in range(iters):
        xs, _ = _rollout(u, x0)
        # Riccati backward pass on linearized dynamics
        Vx = np.array([2 * (xs[-1][0] - 1.0), 0.02 * xs[-1][1]])
        Vxx = np.array([[2.0, 0.0], [0.0, 0.02]])
        k, K = np.zeros(HORIZON), np.zeros((HORIZON, 2))
        for t in range(HORIZON - 1, -1, -1):
            x = xs[t]
            A = np.array([[1.0, DT * (1 + 0.4 * x[1])], [0.0, 1.0]])
            B = np.array([0.0, DT * (1 - np.tanh(u[t]) ** 2)])
            qx = np.array([2 * (x[0] - 1.0), 0.02 * x[1]])
            qu = 0.002 * u[t]
            Qx = np.array([[2.0, 0.0], [0.0, 0.02]])
            Qu = np.array([[0.002]])
            gu = float(qu + B @ Vx)
            gx = qx + A.T @ Vx
            Guu = float(Qu[0, 0] + B @ Vxx @ B)
            Gux = B @ Vxx @ A
            Gxx = Qx + A.T @ Vxx @ A
            k[t] = -gu / Guu
            K[t] = -Gux / Guu
            Vx = gx + Gux * k[t] + K[t] * gu + K[t] * Guu * k[t]
            Vxx = Gxx + np.outer(K[t], Gux) + np.outer(Gux, K[t]) + np.outer(K[t], K[t]) * Guu
        # forward pass
        xs_new = [x0.copy()]
        x = x0.copy()
        for t in range(HORIZON):
            du = k[t] + K[t] @ (x - xs[t])
            u[t] = np.clip(u[t] + du, -3, 3)
            x = _dyn(x, u[t])
            xs_new.append(x.copy())
    _, tot = _rollout(u, x0)
    _, cost_b = pd_baseline(x0, 1.0)
    return {
        "synthetic_ddp_cost": float(tot),
        "synthetic_pd_cost": float(cost_b),
        "synthetic_ddp_gain": float(cost_b - tot),
        "synthetic_torch_available": 0.0,
    }
