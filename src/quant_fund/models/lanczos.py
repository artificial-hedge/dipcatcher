"""Lanczos canon: tridiagonalization of a symmetric matrix
via three-term Krylov recurrence, with optional selective
reorthogonalization, recovering top Ritz eigenvalues on a
synthetic sparse SPD operator.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def lanczos(
    matvec,
    n: int,
    steps: int,
    rng: np.random.Generator,
    reorth: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Run Lanczos; returns (alphas, betas) of the tridiagonal."""
    q = rng.standard_normal(n)
    q /= np.linalg.norm(q)
    qs = [q]
    alphas = np.zeros(steps)
    betas = np.zeros(max(steps - 1, 0))
    q_prev = np.zeros(n)
    beta_prev = 0.0
    for j in range(steps):
        z = np.asarray(matvec(q), dtype=np.float64) - beta_prev * q_prev
        alphas[j] = float(q @ z)
        z -= alphas[j] * q
        if reorth:
            for qi in qs:
                z -= (z @ qi) * qi
        beta = float(np.linalg.norm(z))
        if j < steps - 1:
            if beta < 1e-12:
                return alphas[: j + 1], betas[:j]
            betas[j] = beta
            q_prev, q = q, z / beta
            qs.append(q)
            beta_prev = beta
    return alphas, betas


def ritz_values(alphas: FloatArray, betas: FloatArray, k: int) -> FloatArray:
    """Top-k Ritz eigenvalues of the tridiagonal Jacobi matrix."""
    m = len(alphas)
    t = np.diag(alphas)
    for i in range(m - 1):
        t[i, i + 1] = t[i + 1, i] = betas[i]
    ev = np.linalg.eigvalsh(t)
    return ev[-k:][::-1]


def bench_lanczos(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 400
    # sparse SPD with clustered spectrum
    g = rng.standard_normal((n, 60)) / np.sqrt(60)
    a = g @ g.T + 0.05 * np.eye(n)
    a += 4.0 * np.eye(n)[:n, :n] * 0.0  # no-op guard
    lam_true = np.linalg.eigvalsh(a)[-6:][::-1]

    def mv(x: FloatArray) -> FloatArray:
        return np.asarray(a @ x, dtype=np.float64)

    steps = 60
    al, be = lanczos(mv, n, steps, rng, reorth=True)
    ritz = ritz_values(al, be, 6)
    err = float(np.max(np.abs(ritz - lam_true) / np.maximum(np.abs(lam_true), 1e-9)))
    # orthogonality of Lanczos basis under reorth
    al2, be2 = lanczos(mv, n, steps, rng, reorth=False)
    ritz2 = ritz_values(al2, be2, 6)
    err2 = float(np.max(np.abs(ritz2 - lam_true) / np.maximum(np.abs(lam_true), 1e-9)))
    return {
        "synthetic_ritz_err_reorth": err,
        "synthetic_ritz_err_plain": err2,
        "synthetic_top_eig": float(ritz[0]),
    }
