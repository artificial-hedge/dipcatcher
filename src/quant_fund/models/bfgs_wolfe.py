"""BFGS with strong-Wolfe line search (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _wolfe(f, g, x: np.ndarray, p: np.ndarray, c1=1e-4, c2=0.9) -> float:
    alpha = 1.0
    f0 = f(x)
    g0 = g(x) @ p
    for _ in range(40):
        xa = x + alpha * p
        if f(xa) > f0 + c1 * alpha * g0:
            alpha *= 0.5
            continue
        if abs(g(xa) @ p) <= c2 * abs(g0):
            return alpha
        if (g(xa) @ p) < 0:
            alpha *= 1.5
        else:
            return alpha
    return alpha


def bfgs(f, g, x0: np.ndarray, iters: int = 200) -> np.ndarray:
    x = x0.copy()
    n = len(x)
    b = np.eye(n)
    for _ in range(iters):
        gx = g(x)
        if np.linalg.norm(gx) < 1e-8:
            break
        p = -b @ gx
        if p @ gx >= 0:
            b = np.eye(n)
            p = -gx
        a = _wolfe(f, g, x, p)
        xn = x + a * p
        s = xn - x
        y = g(xn) - gx
        sy = s @ y
        if sy > 1e-12:
            rho = 1.0 / sy
            i1 = np.eye(n)
            b = (i1 - rho * np.outer(s, y)) @ b @ (i1 - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = xn
    return x


def _bench_bfgs_wolfe(seed: int = 0) -> float:
    checks = []
    # quadratic
    q = np.array([[4.0, 1.0], [1.0, 2.0]])
    c = np.array([-1.0, -3.0])
    f = lambda v: float(0.5 * v @ q @ v + c @ v)  # noqa: E731
    g = lambda v: q @ v + c  # noqa: E731
    x = bfgs(f, g, np.array([3.0, 3.0]))
    xstar = -np.linalg.solve(q, c)
    checks.append(np.allclose(x, xstar, atol=1e-5))
    # Rosenbrock
    fr = lambda v: float((1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2)  # noqa: E731
    gr = lambda v: np.array(  # noqa: E731
        [
            -2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2),
            200 * (v[1] - v[0] ** 2),
        ]
    )
    xr = bfgs(fr, gr, np.array([-1.2, 1.0]), iters=400)
    checks.append(np.linalg.norm(xr - np.array([1.0, 1.0])) < 0.05)
    # 4-D quadratic
    q4 = np.diag([1.0, 2.0, 4.0, 8.0])
    c4 = np.array([1.0, -2.0, 3.0, -4.0])
    f4 = lambda v: float(0.5 * v @ q4 @ v + c4 @ v)  # noqa: E731
    g4 = lambda v: q4 @ v + c4  # noqa: E731
    x4 = bfgs(f4, g4, np.zeros(4))
    checks.append(np.allclose(x4, -np.linalg.solve(q4, c4), atol=1e-5))
    # gradient at optimum ~0
    checks.append(np.linalg.norm(g(x)) < 1e-5)
    # superlinear: converged value optimal
    checks.append(abs(fr(xr)) < 1e-8)
    return float(sum(checks) / len(checks))


def bench_bfgs_wolfe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bfgs_wolfe": _bench_bfgs_wolfe(seed)}
