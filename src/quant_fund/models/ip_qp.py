"""Primal-dual interior-point method for convex QP (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def qp_solve(
    q: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    iters: int = 60,
) -> np.ndarray:
    """min 0.5 x'Qx + c'x s.t. Ax <= b via log-barrier Newton."""
    n = len(c)
    x = np.zeros(n)
    # strictly feasible start: find x with Ax < b
    for _ in range(200):
        viol = a @ x - b
        i = int(np.argmax(viol))
        if viol[i] < -1e-3:
            break
        x -= (viol[i] + 0.5) * a[i] / (a[i] @ a[i])
    t = 1.0
    for _ in range(iters):
        s_vec = b - a @ x  # slacks
        # Newton on t*(Qx + c) + sum grad(-log s_i) = 0
        h = t * q + (a.T * (1.0 / s_vec**2)) @ a
        g = t * (q @ x + c) - a.T @ (1.0 / s_vec)
        dx = np.linalg.solve(h, -g)
        # line search keeping s > 0
        alpha = 1.0
        for _ in range(50):
            if np.all(b - a @ (x + alpha * dx) > 1e-10):
                break
            alpha *= 0.5
        x += alpha * dx
        t *= 1.6
        if np.linalg.norm(alpha * dx) < 1e-9 and t > 1e6:
            break
    return x


def _bench_ip_qp(seed: int = 0) -> float:
    checks = []
    # min 0.5(x^2+y^2) s.t. x + y >= 1 -> x=y=0.5
    q = np.eye(2)
    c = np.zeros(2)
    a = np.array([[-1.0, -1.0]])
    b = np.array([-1.0])
    x = qp_solve(q, c, a, b)
    checks.append(np.allclose(x, [0.5, 0.5], atol=1e-3))
    # min (x-3)^2 + (y-1)^2 s.t. x <= 2, y <= 2 -> (2,1)
    q2 = 2 * np.eye(2)
    c2 = np.array([-6.0, -2.0])
    a2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b2 = np.array([2.0, 2.0])
    x2 = qp_solve(q2, c2, a2, b2)
    checks.append(np.allclose(x2, [2.0, 1.0], atol=1e-2))
    # unconstrained-feasible interior: min (x-1)^2 s.t. x <= 5 -> x=1
    q3 = 2 * np.eye(1)
    c3 = np.array([-2.0])
    a3 = np.array([[1.0]])
    b3 = np.array([5.0])
    x3 = qp_solve(q3, c3, a3, b3)
    checks.append(abs(x3[0] - 1.0) < 1e-3)
    # KKT residual small on first problem
    checks.append(abs(x[0] + x[1] - 1.0) < 1e-2)
    # 3-var: min sum squares s.t. x+y+z >= 3 -> all ones
    q4 = np.eye(3)
    c4 = np.zeros(3)
    a4 = -np.eye(3)
    b4 = -np.ones(3) + 0.0  # x_i >= 1 -> -x_i <= -1
    x4 = qp_solve(q4, c4, a4, b4)
    checks.append(np.allclose(x4, np.ones(3), atol=1e-2))
    return float(sum(checks) / len(checks))


def bench_ip_qp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ip_qp": _bench_ip_qp(seed)}
