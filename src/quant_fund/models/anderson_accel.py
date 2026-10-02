"""Anderson acceleration for fixed-point iteration canon.

Given a contractive-ish map g with fixed point x* = g(x*), plain
Picard iteration x_{k+1} = g(x_k) converges only linearly.
Anderson acceleration mixes the last m iterates with weights
minimizing the combined residual — a multi-secant quasi-Newton
acceleration.

- ``anderson_aa`` — type-II via unconstrained least squares on
  residual differences; optional relaxation (beta) and cap on
  the history m.
- ``picard`` — plain fixed-point baseline.

Bench: a slowly contracting map on R^d where AA cuts the
iteration count dramatically (SYNTHETIC fixture only).
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def picard(
    g,
    x0: FloatArray,
    it: int = 10000,
    tol: float = 1e-10,
) -> tuple[FloatArray, int]:
    x = np.asarray(x0, dtype=np.float64).copy()
    for k in range(1, it + 1):
        xn = np.asarray(g(x), dtype=np.float64)
        if np.linalg.norm(xn - x) < tol:
            return xn, k
        x = xn
    return x, it


def anderson_aa(
    g,
    x0: FloatArray,
    m: int = 5,
    beta: float = 1.0,
    it: int = 5000,
    tol: float = 1e-10,
) -> tuple[FloatArray, int]:
    """Type-II Anderson acceleration with depth-m history."""
    x = np.asarray(x0, dtype=np.float64).copy()
    fs: list[FloatArray] = []  # residuals g(x)-x
    xs: list[FloatArray] = []
    for k in range(1, it + 1):
        gx = np.asarray(g(x), dtype=np.float64)
        fk = gx - x
        if np.linalg.norm(fk) < tol:
            return gx, k
        xs.append(x)
        fs.append(fk)
        if len(fs) > m:
            xs.pop(0)
            fs.pop(0)
        mk = len(fs)
        if mk == 1:
            x = x + beta * fk
            continue
        # Solve min ||sum alpha_j (f_k - f_{k-j-1})|| style: type-II
        # formulation solves min_gamma ||f_k - D_k gamma|| with
        # D_k[:,j] = f_k - f_{k-j-1}.
        dmat = np.column_stack([fs[-1] - fs[-1 - j] for j in range(1, mk)])
        gamma, *_ = np.linalg.lstsq(dmat, fs[-1], rcond=None)
        dx = np.column_stack([xs[-1] - xs[-1 - j] for j in range(1, mk)])
        x = np.asarray(
            xs[-1] + beta * fs[-1] - (dx + beta * dmat) @ gamma,
            dtype=np.float64,
        )
        if not np.all(np.isfinite(x)):
            return x, k
    return x, it


def bench_anderson(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = 30
    # g(x) = A x + b with contraction factor ~0.97 — Picard crawls.
    # Fixed point of a small-step gradient map: g(x) = x - e·Mx.
    m_pos = rng.standard_normal((d, d))
    m_pos = m_pos.T @ m_pos + 0.5 * np.eye(d)
    x_star = rng.standard_normal(d)
    c = m_pos @ x_star
    lam_max = float(np.linalg.eigvalsh(m_pos).max())
    eps = 1.9 / lam_max  # near-nonexpansive: slow contraction

    def g(x: FloatArray) -> FloatArray:
        return np.asarray(x - eps * (m_pos @ x - c), dtype=np.float64)

    x0 = np.zeros(d)
    xp, n_pic = picard(g, x0, tol=1e-10)
    xa, n_aa = anderson_aa(g, x0, m=8, tol=1e-10)
    err_p = float(np.linalg.norm(xp - x_star))
    err_a = float(np.linalg.norm(xa - x_star))
    return {
        "synthetic_picard_iters": float(n_pic),
        "synthetic_aa_iters": float(n_aa),
        "synthetic_aa_err": err_a,
        "synthetic_picard_err": err_p,
    }


__all__ = ["anderson_aa", "bench_anderson", "picard"]
