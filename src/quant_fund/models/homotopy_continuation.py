"""Homotopy continuation / path-following canon.

Solving F(x)=0 by embedding it in H(x,t) = t·F(x) + (1-t)·G(x)
where G is a trivial system with known root x0. Three drivers:

- ``newton_homotopy`` — plain parameter stepping: at each t_k,
  re-solve H(·,t_k)=0 by Newton from the previous solution.
- ``fixed_newton`` — Newton on F directly (baseline that often
  diverges from a bad start).
- ``pc_continuation`` — Euler predictor along the solution curve
  (dx/dt = -H_x⁻¹ H_t) followed by Newton corrector at the new t.

Generic capability for robust root-finding when good initial
guesses are unavailable; benches are SYNTHETIC fixtures only.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def _newton_step(f, jac, x: FloatArray, tol: float) -> FloatArray:
    """Damped Newton step: Armijo backtracking on ||f||^2 / 2."""
    r = np.asarray(f(x), dtype=np.float64)
    if np.linalg.norm(r) < tol:
        return x
    j = np.asarray(jac(x), dtype=np.float64)
    step = np.linalg.solve(j, r)
    alpha = 1.0
    nrm0 = float(r @ r)
    for _ in range(25):
        xn = x - alpha * step
        if np.all(np.isfinite(xn)):
            rn = np.asarray(f(xn), dtype=np.float64)
            if np.all(np.isfinite(rn)) and float(rn @ rn) <= (1 - 1e-4 * alpha) * nrm0:
                return np.asarray(xn, dtype=np.float64)
        alpha *= 0.5
    return np.asarray(x - alpha * step, dtype=np.float64)


def newton_solve(
    f,
    jac,
    x0: FloatArray,
    it: int = 100,
    tol: float = 1e-12,
) -> tuple[FloatArray, bool]:
    """Plain Newton with failure flag."""
    x = np.asarray(x0, dtype=np.float64).copy()
    for _ in range(it):
        x = _newton_step(f, jac, x, tol)
        if not np.all(np.isfinite(x)):
            return np.asarray(x, dtype=np.float64), False
        if np.linalg.norm(np.asarray(f(x), dtype=np.float64)) < tol:
            return np.asarray(x, dtype=np.float64), True
    return np.asarray(x, dtype=np.float64), bool(
        np.linalg.norm(np.asarray(f(x), dtype=np.float64)) < tol
    )


def newton_homotopy(
    f,
    jac,
    g0: FloatArray,
    x0: FloatArray,
    steps: int = 10,
    tol: float = 1e-12,
) -> tuple[FloatArray, int]:
    """Parameter stepping on H(x,t) = t·F + (1-t)·(x - x0); g0 unused.

    Returns (root, total Newton iterations).
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    n_it = 0
    for k in range(1, steps + 1):
        t = k / steps

        def h(z: FloatArray, t: float = t) -> FloatArray:
            return np.asarray(
                t * np.asarray(f(z), dtype=np.float64) + (1 - t) * (z - x0),
                dtype=np.float64,
            )

        def hj(z: FloatArray, t: float = t) -> FloatArray:
            return np.asarray(
                t * np.asarray(jac(z), dtype=np.float64) + (1 - t) * np.eye(len(z)),
                dtype=np.float64,
            )

        for _ in range(60):
            x = _newton_step(h, hj, x, tol)
            n_it += 1
            if np.linalg.norm(np.asarray(h(x))) < tol:
                break
    return np.asarray(x, dtype=np.float64), n_it


def pc_continuation(
    f,
    jac,
    x0: FloatArray,
    steps: int = 40,
    tol: float = 1e-12,
) -> tuple[FloatArray, int]:
    """Euler-Newton predictor-corrector on H = t·F + (1-t)·(x-x0)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    n_it = 0
    t = 0.0
    dt = 1.0 / steps
    while t < 1.0 - 1e-15:
        t = min(1.0, t + dt)
        jx = t * np.asarray(jac(x), dtype=np.float64) + (1 - t) * np.eye(len(x))
        ht = np.asarray(f(x), dtype=np.float64) - (x - x0)
        x = x + dt * np.linalg.solve(jx, -ht)  # Euler predictor
        for _ in range(50):  # damped Newton corrector

            def h(z: FloatArray, t: float = t) -> FloatArray:
                return np.asarray(
                    t * np.asarray(f(z), dtype=np.float64) + (1 - t) * (z - x0),
                    dtype=np.float64,
                )

            def hj(z: FloatArray, t: float = t) -> FloatArray:
                return np.asarray(
                    t * np.asarray(jac(z), dtype=np.float64) + (1 - t) * np.eye(len(z)),
                    dtype=np.float64,
                )

            if np.linalg.norm(h(x)) < tol:
                break
            x = _newton_step(h, hj, x, tol)
            n_it += 1
    return np.asarray(x, dtype=np.float64), n_it


def bench_homotopy(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = 6
    # Planted root x*; F(x) = A·x - b + nonlinearity, scaled so that
    # plain Newton from origin diverges/stalls.
    a = rng.standard_normal((d, d)) + 4.0 * np.eye(d)
    x_star = rng.standard_normal(d)
    c = 2.0

    def f(x: FloatArray) -> FloatArray:
        return np.asarray(
            a @ x - a @ x_star + c * np.sin(3.0 * (x - x_star)),
            dtype=np.float64,
        )

    def jac(x: FloatArray) -> FloatArray:
        return np.asarray(
            a + 3.0 * c * np.diag(np.cos(3.0 * (x - x_star))),
            dtype=np.float64,
        )

    x0 = x_star + 8.0 * rng.standard_normal(d)
    xn, ok = newton_solve(f, jac, x0, it=30)
    err_n = float(np.linalg.norm(xn - x_star)) if ok else 50.0
    xh, _ = newton_homotopy(f, jac, x0, x0, steps=12)
    xp, _ = pc_continuation(f, jac, x0, steps=30)
    err_h = float(np.linalg.norm(xh - x_star))
    err_p = float(np.linalg.norm(xp - x_star))
    return {
        "synthetic_newton_err": err_n,
        "synthetic_homotopy_err": err_h,
        "synthetic_pc_err": err_p,
    }


__all__ = [
    "bench_homotopy",
    "newton_solve",
    "newton_homotopy",
    "pc_continuation",
]
