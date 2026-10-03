"""Sequential quadratic programming for equality-constrained NLP (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def sqp_solve(f, grad_f, h, jac_h, x0: np.ndarray, iters: int = 60) -> np.ndarray:
    """Newton-type SQP on the KKT system: solves min f s.t. h(x)=0.

    Each step solves the equality QP with the Lagrangian Hessian
    L_xx = grad^2 f + sum_k lam_k grad^2 h_k:

      [H  J^T][dx] = [-g - J^T lam]
      [J   0 ][dl]   [-h]
    """
    x = x0.copy()
    m = len(h(x0))
    lam = np.zeros(m)
    for _ in range(iters):
        g = grad_f(x)
        j = jac_h(x)
        hv = h(x)
        n = len(x)
        hmat = _hess_fd(grad_f, x)
        for k in range(m):
            hmat += lam[k] * _hess_fd(lambda v, kk=k: jac_h(v)[kk, :], x)
        kkt = np.zeros((n + m, n + m))
        kkt[:n, :n] = hmat + 1e-10 * np.eye(n)
        kkt[:n, n:] = j.T
        kkt[n:, :n] = j
        rhs = np.concatenate([-g - j.T @ lam, -hv])
        try:
            d = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            break
        # damped update for safety
        x += d[:n]
        lam += d[n:]
        if np.linalg.norm(d[:n]) < 1e-10:
            break
    return x


def _hess_fd(g, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = len(x)
    h = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        h[:, i] = (g(x + e) - g(x - e)) / (2 * eps)
    return (h + h.T) / 2


def _bench_sqp(seed: int = 0) -> float:
    checks = []
    # min x^2 + y^2 s.t. x + y = 1 -> x = y = 0.5
    f = lambda v: float(v[0] ** 2 + v[1] ** 2)  # noqa: E731
    g = lambda v: np.array([2 * v[0], 2 * v[1]])  # noqa: E731
    h = lambda v: np.array([v[0] + v[1] - 1.0])  # noqa: E731
    j = lambda v: np.array([[1.0, 1.0]])  # noqa: E731
    x = sqp_solve(f, g, h, j, np.array([0.2, 0.9]))
    checks.append(np.allclose(x, [0.5, 0.5], atol=1e-6))
    # min (x-2)^2 + y^2 s.t. y = x^2: constraint curve
    f2 = lambda v: float((v[0] - 2) ** 2 + v[1] ** 2)  # noqa: E731
    g2 = lambda v: np.array([2 * (v[0] - 2), 2 * v[1]])  # noqa: E731
    h2 = lambda v: np.array([v[1] - v[0] ** 2])  # noqa: E731
    j2 = lambda v: np.array([[-2 * v[0], 1.0]])  # noqa: E731
    x2 = sqp_solve(f2, g2, h2, j2, np.array([0.5, 0.5]))
    checks.append(abs(h2(x2)[0]) < 1e-8)
    # optimum on parabola closest to (2,0): root of 4x^3 + 2x - 4 = 0
    checks.append(abs(x2[0] - 0.8351223484813665) < 1e-4)
    # constraint residual goes to 0
    checks.append(abs(h(x)[0]) < 1e-9)
    # min on circle x^2+y^2=1 of linear 2x+3y -> -(2,3)/sqrt13 direction
    f3 = lambda v: float(2 * v[0] + 3 * v[1])  # noqa: E731
    g3 = lambda v: np.array([2.0, 3.0])  # noqa: E731
    h3 = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 1.0])  # noqa: E731
    j3 = lambda v: np.array([[2 * v[0], 2 * v[1]]])  # noqa: E731
    x3 = sqp_solve(f3, g3, h3, j3, np.array([-0.5, -0.8]))
    opt = -np.array([2.0, 3.0]) / np.sqrt(13.0)
    checks.append(np.allclose(x3, opt, atol=1e-3))
    return float(sum(checks) / len(checks))


def bench_sqp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sqp": _bench_sqp(seed)}
