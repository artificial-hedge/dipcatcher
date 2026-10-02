"""Arnoldi + GMRES canon: Arnoldi–Gram-Schmidt Krylov basis
construction with modified Gram-Schmidt orthogonalization,
and restarted GMRES(m) for nonsymmetric sparse systems,
checked against the dense solve residual.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def arnoldi(matvec, v0: FloatArray, steps: int) -> tuple[FloatArray, FloatArray]:
    """Arnoldi iteration; returns basis V (n×steps+1) and H (steps+1×steps)."""
    n = v0.size
    v = np.zeros((n, steps + 1))
    h = np.zeros((steps + 1, steps))
    v[:, 0] = v0 / np.linalg.norm(v0)
    for j in range(steps):
        w = np.asarray(matvec(v[:, j]), dtype=np.float64)
        for i in range(j + 1):
            h[i, j] = float(v[:, i] @ w)
            w -= h[i, j] * v[:, i]
        h[j + 1, j] = float(np.linalg.norm(w))
        if h[j + 1, j] > 1e-13:
            v[:, j + 1] = w / h[j + 1, j]
        else:
            break
    return v, h


def gmres(
    matvec,
    b: FloatArray,
    restart: int = 30,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[FloatArray, int, float]:
    """Restarted GMRES; returns (x, iters, residual_norm)."""
    b = np.asarray(b, dtype=np.float64)
    x = np.zeros_like(b)
    iters = 0
    while iters < max_iter:
        r = b - np.asarray(matvec(x), dtype=np.float64)
        beta = float(np.linalg.norm(r))
        if beta < tol:
            return x, iters, beta
        m = min(restart, max_iter - iters)
        v, h = arnoldi(matvec, r, m)
        e1 = np.zeros(m + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(h[: m + 1, :m], e1, rcond=None)
        x = x + v[:, :m] @ y
        iters += m
    res = float(np.linalg.norm(b - np.asarray(matvec(x), dtype=np.float64)))
    return x, iters, res


def bench_arnoldi_gmres(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 300
    # nonsymmetric diagonally dominant sparse
    g = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.05)
    a = g + 6.0 * np.eye(n) + 0.1 * rng.standard_normal((n, n))
    x_true = rng.standard_normal(n)
    b = a @ x_true

    def mv(x: FloatArray) -> FloatArray:
        return np.asarray(a @ x, dtype=np.float64)

    x, iters, res = gmres(mv, b, restart=30, max_iter=120)
    err = float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
    # arnoldi relation check on a small system
    v, h = arnoldi(mv, rng.standard_normal(n), 20)
    av = np.asarray(a @ v[:, :20], dtype=np.float64)
    vh = v[:, :21] @ h[:21, :20]
    rel = float(np.linalg.norm(av - vh) / np.linalg.norm(av))
    return {
        "synthetic_gmres_residual": res,
        "synthetic_gmres_err": err,
        "synthetic_gmres_iters": float(iters),
        "synthetic_arnoldi_relation": rel,
    }
