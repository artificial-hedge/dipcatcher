"""ILU(0) preconditioned conjugate gradient for SPD matrices (SYNTHETIC).

ILU(0): incomplete LU keeping only the original sparsity pattern —
lower-triangular forward/backward substitutions apply the
preconditioner. On a 2-D Poisson matrix, PCG converges much faster than
raw CG. Verified: iteration reduction factor and final accuracy vs the
dense solve.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 968


def ilu0(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Doolittle ILU restricted to the sparsity pattern of A."""
    n = a.shape[0]
    lo = np.eye(n)
    u = np.zeros_like(a)
    pattern = a != 0
    for i in range(n):
        for j in range(n):
            if not pattern[i, j]:
                continue
            s = float(a[i, j] - (lo[i, : min(i, j)] * u[: min(i, j), j]).sum())
            if i <= j:
                u[i, j] = s
            else:
                lo[i, j] = s / u[j, j]
    return lo, u


def solve_lu(lo: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - lo[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - u[i, i + 1 :] @ x[i + 1 :]) / u[i, i]
    return x


def pcg(
    a: np.ndarray,
    b: np.ndarray,
    precond: tuple[np.ndarray, np.ndarray] | None = None,
    tol: float = 1e-10,
    maxit: int = 500,
) -> tuple[np.ndarray, int]:
    x = np.zeros_like(b)
    r = b - a @ x
    if precond is None:
        z = r.copy()
        apply = lambda v: v  # noqa: E731
    else:
        lo, u = precond
        apply = lambda v: solve_lu(lo, u, v)  # noqa: E731
        z = apply(r)
    p = z.copy()
    rz = float(r @ z)
    it = 0
    for it in range(1, maxit + 1):  # noqa: B007
        ap = a @ p
        alpha = rz / float(p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        if np.linalg.norm(r) < tol * np.linalg.norm(b):
            break
        z = apply(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it


def poisson_2d_n(g: int) -> np.ndarray:
    n = g * g
    a = 4.0 * np.eye(n)
    for i in range(g):
        for j in range(g):
            k = i * g + j
            if i > 0:
                a[k, k - g] = -1.0
            if i < g - 1:
                a[k, k + g] = -1.0
            if j > 0:
                a[k, k - 1] = -1.0
            if j < g - 1:
                a[k, k + 1] = -1.0
    return a


def bench_ilu_precond(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    g = 12
    a = poisson_2d_n(g)
    b = rng.normal(size=g * g)
    exact = np.linalg.solve(a, b)
    xp, it_p = pcg(a, b, precond=ilu0(a))
    xc, it_c = pcg(a, b)
    err = float(np.linalg.norm(xp - exact) / np.linalg.norm(exact))
    checks = [
        err < 1e-8,
        it_p < it_c,  # preconditioned CG converges in fewer iters
        it_p <= it_c // 2,
        it_c > 20,
    ]
    return {"synthetic_ilu_precond": float(np.mean(checks))}
