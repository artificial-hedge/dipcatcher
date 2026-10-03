"""Lyapunov functions: V>0, Vdot<=0 => stability (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def v_quad(x: np.ndarray, p: np.ndarray) -> float:
    """V(x) = x' P x."""
    return float(x @ p @ x)


def vdot_quadratic(x: np.ndarray, a: np.ndarray, p: np.ndarray) -> float:
    """Vdot = grad V . f = x' (A'P + PA) x."""
    return float(x @ (a.T @ p + p @ a) @ x)


def is_pos_def(m: np.ndarray) -> bool:
    return bool(np.all(np.linalg.eigvalsh(m) > 0))


def _bench_lyapunov_stability(seed: int = 0) -> float:
    checks = []
    # stable spiral a = [[-1,-2],[2,-1]], V = x'x
    a = np.array([[-1.0, -2.0], [2.0, -1.0]])
    p = np.eye(2)
    # A' + A = [[-2,0],[0,-2]] < 0
    checks.append(is_pos_def(-(a.T @ p + p @ a)))
    x = np.array([1.0, 0.5])
    checks.append(v_quad(x, p) > 0 and vdot_quadratic(x, a, p) < 0)
    # solve Lyapunov eq A'P + PA = -I for P
    # vectorize: solve kron system
    n = 2
    k = np.kron(np.eye(n), a.T) + np.kron(a.T, np.eye(n))
    pv = np.linalg.solve(k, -np.eye(n).flatten())
    ps = pv.reshape(n, n)
    checks.append(is_pos_def(ps))
    checks.append(np.allclose(a.T @ ps + ps @ a, -np.eye(n)))
    # unstable system can't have this V decreasing everywhere
    au = np.array([[1.0, 0.0], [0.0, 1.0]])
    checks.append(vdot_quadratic(x, au, p) > 0)
    return float(sum(checks) / len(checks))


def bench_lyapunov_stability(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lyapunov_stability": _bench_lyapunov_stability(seed)}
