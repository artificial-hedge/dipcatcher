"""Interpolative decomposition canon (Cheng, Gimbutas, (SYNTHETIC)
Martinsson & Rokhlin 2005): A ≈ B P where B is k actual
columns of A selected by column-pivoted QR (Businger-Golub),
and P contains a k×k identity block.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def pivoted_qr(a: FloatArray, k: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Column-pivoted QR (Businger-Golub); returns Q, R, permutation."""
    a = np.asarray(a, dtype=np.float64)
    m, n = a.shape
    k = min(k, min(m, n))
    r = a.copy()
    perm = np.arange(n)
    q = np.eye(m)
    for j in range(k):
        norms = np.linalg.norm(r[j:, j:], axis=0)
        piv = int(np.argmax(norms)) + j
        r[:, [j, piv]] = r[:, [piv, j]]
        perm[[j, piv]] = perm[[piv, j]]
        x = r[j:, j].copy()
        nx = np.linalg.norm(x)
        if nx < 1e-13:
            continue
        e1 = np.zeros_like(x)
        e1[0] = -np.sign(x[0]) * nx if x[0] != 0 else -nx
        v = x - e1
        v /= np.linalg.norm(v)
        hj = np.eye(m - j) - 2.0 * np.outer(v, v)
        r[j:, j:] = hj @ r[j:, j:]
        q[:, j:] = q[:, j:] @ hj
    return q, r, perm


def interpolative_decomp(a: FloatArray, k: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """ID: A ≈ B P, B = A[:, skel], P = [I | T] under permutation."""
    a = np.asarray(a, dtype=np.float64)
    _, r, perm = pivoted_qr(a, k)
    r11 = r[:k, :k]
    r12 = r[:k, k:]
    t = np.linalg.solve(r11, r12)
    skel = perm[:k]
    b = a[:, skel]
    p = np.zeros((k, a.shape[1]))
    p[:, perm[:k]] = np.eye(k)
    p[:, perm[k:]] = t
    return b, p, skel


def bench_interpolative_decomp(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    m, n, r = 250, 200, 10
    u0, _ = np.linalg.qr(rng.standard_normal((m, r)))
    v0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    a = u0 @ np.diag(np.geomspace(30.0, 0.4, r)) @ v0.T
    a += 0.06 * rng.standard_normal((m, n))
    k = 10
    b, p, skel = interpolative_decomp(a, k)
    err = float(np.linalg.norm(a - b @ p) / np.linalg.norm(a))
    u_ex, s_ex, vt_ex = np.linalg.svd(a, full_matrices=False)
    opt = u_ex[:, :k] @ np.diag(s_ex[:k]) @ vt_ex[:k]
    opt_err = float(np.linalg.norm(a - opt) / np.linalg.norm(a))
    # P embeds a k×k identity on the skeleton columns
    ident_err = float(np.max(np.abs(p[:, skel] - np.eye(k))))
    return {
        "synthetic_id_err": err,
        "synthetic_opt_err": opt_err,
        "synthetic_skeleton_size": float(len(skel)),
        "synthetic_identity_err": ident_err,
    }
