"""Iterative least-squares canon: CGLS and LSQR.

For min_x ||A x - b||^2 without forming A'A explicitly —
matrix-free via ``matvec``/``rmatvec`` callables:

- ``cgls`` — conjugate-gradient applied to the normal equations
  (Craig's method; stable variant of CGNE).
- ``lsqr`` — Paige-Saunders Golub-Kahan bidiagonalization with
  the QR subproblem carried along via Givens rotations.

Bench: an ill-conditioned sparse-design regression where LSQR
matches the dense normal-equations solution (SYNTHETIC only).
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def cgls(
    matvec,
    rmatvec,
    b: FloatArray,
    it: int = 1000,
    tol: float = 1e-10,
) -> tuple[FloatArray, int]:
    b = np.asarray(b, dtype=np.float64)
    n = len(rmatvec(np.zeros_like(b)))
    x = np.zeros(n)
    r = b.copy()
    s = np.asarray(rmatvec(r))
    p = s.copy()
    gamma = float(s @ s)
    for k in range(1, it + 1):
        q = np.asarray(matvec(p))
        dq = float(q @ q)
        if dq <= 0:
            break
        alpha = gamma / dq
        x += alpha * p
        r -= alpha * q
        s = np.asarray(rmatvec(r))
        gamma_new = float(s @ s)
        if gamma_new < tol * tol:
            return x, k
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return x, it


def lsqr(
    matvec,
    rmatvec,
    b: FloatArray,
    it: int = 1000,
    tol: float = 1e-10,
) -> tuple[FloatArray, int]:
    """Paige-Saunders LSQR."""
    b = np.asarray(b, dtype=np.float64)
    beta = np.linalg.norm(b)
    if beta == 0:
        return np.zeros(len(rmatvec(b))), 0
    u = b / beta
    v = np.asarray(rmatvec(u))
    alpha = np.linalg.norm(v)
    if alpha == 0:
        return np.zeros(len(v)), 0
    v = v / alpha
    w = v.copy()
    x = np.zeros(len(v))
    phi_bar, rho_bar = beta, alpha
    c, s_g = 1.0, 0.0
    for k in range(1, it + 1):
        # Golub-Kahan bidiagonalization step
        u = np.asarray(matvec(v)) - alpha * u
        beta = np.linalg.norm(u)
        if beta > 0:
            u = u / beta
        v = np.asarray(rmatvec(u)) - beta * v
        alpha = np.linalg.norm(v)
        if alpha > 0:
            v = v / alpha
        # Givens rotation to zero the subdiagonal
        rho = np.hypot(rho_bar, beta)
        c, s_g = rho_bar / rho, beta / rho
        theta = s_g * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s_g * phi_bar
        # Update x, w
        x += (phi / rho) * w
        w = v - (theta / rho) * w
        # Residual estimate |A'r| = phi_bar*alpha*|c|
        if abs(phi_bar * alpha * c) < tol:
            return x, k
    return x, it


def bench_iterative_ls(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, d = 400, 60
    # Ill-conditioned sparse design
    a = rng.standard_normal((n, d))
    mask = rng.random((n, d)) < 0.7
    a[mask] = 0.0
    u, sv, vt = np.linalg.svd(a, full_matrices=False)
    sv = np.geomspace(1.0, 1e-4, d)  # force conditioning
    a = (u * sv) @ vt
    x_star = rng.standard_normal(d)
    b = a @ x_star + 0.01 * rng.standard_normal(n)

    mv = lambda z: a @ z  # noqa: E731
    rv = lambda z: a.T @ z  # noqa: E731

    x_dense = np.linalg.lstsq(a, b, rcond=None)[0]
    xc, it_c = cgls(mv, rv, b)
    xl, it_l = lsqr(mv, rv, b)
    return {
        "synthetic_cgls_err": float(np.linalg.norm(xc - x_dense)),
        "synthetic_lsqr_err": float(np.linalg.norm(xl - x_dense)),
        "synthetic_cgls_iters": float(it_c),
        "synthetic_lsqr_iters": float(it_l),
        "synthetic_cond": float(sv[0] / sv[-1]),
    }


__all__ = ["bench_iterative_ls", "cgls", "lsqr"]
