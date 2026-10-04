"""Broyden's quasi-Newton method for nonlinear systems (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def broyden(
    f,
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> tuple[np.ndarray, int]:
    x = x0.astype(float)
    fx = f(x)
    b = np.eye(len(x)) * 1.0  # initial Jacobian approx
    for it in range(max_iter):
        if np.linalg.norm(fx) < tol:
            return x, it
        dx = np.linalg.solve(b, -fx)
        x_new = x + dx
        fx_new = f(x_new)
        df = fx_new - fx
        b = b + np.outer(df - b @ dx, dx) / (dx @ dx)
        x, fx = x_new, fx_new
    return x, max_iter


def _bench_broyden(seed: int = 0) -> float:
    checks = []

    def f(v: np.ndarray) -> np.ndarray:
        return np.array([v[0] ** 2 + v[1] ** 2 - 4.0, v[0] * v[1] - 1.0])

    x, iters = broyden(f, np.array([2.0, 0.5]))
    checks.append(np.linalg.norm(f(x)) < 1e-8)
    checks.append(iters < 30)
    # root satisfies constraints: circle + hyperbola -> x^2 = 2 +- sqrt(3)
    checks.append(abs(x[0] ** 2 + x[1] ** 2 - 4.0) < 1e-6)
    # converges from another start
    x2, _ = broyden(f, np.array([0.5, 2.0]))
    checks.append(np.linalg.norm(f(x2)) < 1e-8)
    # 1-d equivalent: x^2 - 2 = 0 -> sqrt(2)
    x3, _ = broyden(lambda v: np.array([v[0] ** 2 - 2.0]), np.array([1.5]))
    checks.append(abs(x3[0] - np.sqrt(2.0)) < 1e-8)
    return float(sum(checks) / len(checks))


def bench_broyden(seed: int = 0) -> dict[str, float]:
    return {"synthetic_broyden": _bench_broyden(seed)}
