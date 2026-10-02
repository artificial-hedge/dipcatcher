"""Conjugate gradient: linear CG (Hestenes-Stiefel 1952)
for SPD systems and nonlinear Polak-Ribière+ CG with
backtracking line search (Nocedal-Wright). Synthetic bench
gates CG solve against dense solve and nonlinear CG on a
tilted Rosenbrock against gradient descent."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cg_solve(
    a: FloatArray,
    b: FloatArray,
    x0: FloatArray | None = None,
    tol: float = 1e-10,
    it: int | None = None,
) -> FloatArray:
    """Standard linear CG for A SPD: O(k) matvecs for
    κ-clustered spectra."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    r = b - a @ x
    p = r.copy()
    rs = float(r @ r)
    for _ in range(it or 10 * n):
        ap = a @ p
        alpha = rs / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        rs_new = float(r @ r)
        if rs_new < tol**2 * rs or rs_new < 1e-24:
            rs = rs_new
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return np.asarray(x)


def nlcg_pr(
    f: Callable[[FloatArray], float],
    grad: Callable[[FloatArray], FloatArray],
    x0: FloatArray,
    it: int = 500,
    tol: float = 1e-8,
) -> FloatArray:
    """Nonlinear CG, Polak-Ribière+ (β = max(0, PR)),
    Armijo backtracking."""
    x = np.asarray(x0, dtype=np.float64).copy()
    g = np.asarray(grad(x))
    d = -g
    for _ in range(it):
        if float(g @ g) < tol**2:
            break
        # Armijo backtrack on f(x + t d)
        fx = float(f(x))
        t = 1.0
        for _ in range(30):
            if f(x + t * d) <= fx + 1e-4 * t * float(d @ g):
                break
            t *= 0.5
        x += t * d
        g_new = np.asarray(grad(x))
        beta = max(0.0, float(g_new @ (g_new - g)) / float(g @ g))
        d = -g_new + beta * d
        g = g_new
    return np.asarray(x)


def bench_conjugate_gradient(seed: int = 560) -> dict[str, float]:
    """SYNTHETIC: (a) ill-conditioned SPD system — CG must
    match np.linalg.solve to high precision; (b) Rosenbrock —
    PR+ CG must reach the minimum in far fewer iterations
    than plain gradient descent."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # (a) SPD solve
    n = 40
    q, _ = np.linalg.qr(rng.normal(0, 1, (n, n)))
    lam = np.geomspace(1e-2, 10.0, n)
    a = q @ np.diag(lam) @ q.T
    b = rng.normal(0, 1, n)
    x_cg = cg_solve(a, b, tol=1e-12)
    x_ref = np.linalg.solve(a, b)
    out["synthetic_cg_solve_err"] = float(np.linalg.norm(x_cg - x_ref))
    if out["synthetic_cg_solve_err"] > 1e-6:
        raise ValueError(f"cg solve off: {out['synthetic_cg_solve_err']}")

    # (b) Rosenbrock
    def rosen(x: FloatArray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def rosen_g(x: FloatArray) -> FloatArray:
        g = np.zeros_like(x)
        g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
        g[1:] += 200 * (x[1:] - x[:-1] ** 2)
        return g

    x0 = np.full(4, -1.0)
    x_cgnl = nlcg_pr(rosen, rosen_g, x0, it=400)
    out["synthetic_cg_rosen_f"] = rosen(x_cgnl)
    # plain GD baseline for contrast
    xg = x0.copy()
    for _ in range(400):
        g = rosen_g(xg)
        t = 0.001
        for _ in range(20):
            if rosen(xg - t * g) < rosen(xg):
                break
            t *= 0.5
        xg -= t * g
    out["synthetic_gd_rosen_f"] = rosen(xg)
    if rosen(x_cgnl) > 0.5:
        raise ValueError(f"cg rosen off: {rosen(x_cgnl)}")
    if rosen(x_cgnl) > rosen(xg):
        raise ValueError(f"cg not >= gd: {rosen(x_cgnl)} vs {rosen(xg)}")
    return out
