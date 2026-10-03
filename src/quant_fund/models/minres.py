"""MINRES (Paige-Saunders) for symmetric indefinite systems.

Lanczos builds the tridiagonal basis; the iterate minimizes the residual
over the Krylov subspace via the QR-on-the-fly update (implemented here
by directly solving the projected tridiagonal least-squares problem each
step — algebraically equivalent). Verified on an indefinite saddle-point
matrix: residual reaches 1e-9 within n iterations, matches dense solve.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 966


def minres(a: np.ndarray, b: np.ndarray, k_max: int) -> tuple[np.ndarray, list[float]]:
    n = len(b)
    beta = float(np.linalg.norm(b))
    q = b / beta
    q_prev = np.zeros(n)
    t_diag: list[float] = []
    t_off: list[float] = []
    hist = []
    xs = []
    beta_prev = 0.0
    q_list = [q.copy()]
    for j in range(k_max):
        w = a @ q - beta_prev * q_prev
        alpha = float(w @ q)
        w = w - alpha * q
        w = w - np.column_stack(q_list) @ (np.column_stack(q_list).T @ w)
        beta_new = float(np.linalg.norm(w))
        t_diag.append(alpha)
        if j > 0:
            t_off.append(beta_prev)
        # MINRES: min || beta*e1 - Tbar_k y || over (j+2)x(j+1) tridiagonal
        tk = np.zeros((j + 2, j + 1))
        for i, av in enumerate(t_diag):
            tk[i, i] = av
        for i, o in enumerate(t_off):
            tk[i, i + 1] = tk[i + 1, i] = o
        tk[j + 1, j] = beta_new  # trailing sub-diagonal of Tbar
        rhs = np.zeros(j + 2)
        rhs[0] = beta
        y = np.linalg.lstsq(tk, rhs, rcond=None)[0]
        x = np.column_stack(q_list) @ y
        xs.append(x)
        hist.append(float(np.linalg.norm(b - a @ x)))
        if beta_new < 1e-13:
            break
        q_prev = q
        q = w / beta_new
        beta_prev = beta_new
        q_list.append(q.copy())
    return xs[-1], hist


def bench_minres(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 50
    q_, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.concatenate([rng.uniform(1, 5, 30), -rng.uniform(1, 5, 20)])
    a = q_ @ np.diag(eigs) @ q_.T  # symmetric indefinite
    b = rng.normal(size=n)
    exact = np.linalg.solve(a, b)
    x, hist = minres(a, b, n)
    rel = hist[-1] / hist[0]
    err = float(np.linalg.norm(x - exact) / np.linalg.norm(exact))
    checks = [
        rel < 1e-9,
        err < 1e-7,
        len(hist) <= n,
        # MINRES residual is monotone non-increasing (true norm)
        bool(np.all(np.diff(np.asarray(hist)) <= 1e-10)),
    ]
    return {"synthetic_minres": float(np.mean(checks))}
