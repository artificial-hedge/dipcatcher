"""Adjoint operator: <Ax, y> = <x, A* y> and adjoint algebra (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_adjoint_op(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    a = rng.normal(size=(4, 4))
    x = rng.normal(size=(200, 4))
    y = rng.normal(size=(200, 4))
    lhs = np.sum((x @ a.T) * y, axis=1)
    rhs = np.sum(x * (y @ a), axis=1)
    checks.append(np.allclose(lhs, rhs))
    # (AB)* = B* A*
    b = rng.normal(size=(4, 4))
    lhs2 = np.sum((x @ (a @ b).T) * y, axis=1)
    rhs2 = np.sum(x * (y @ (b.T @ a.T).T), axis=1)
    checks.append(np.allclose(lhs2, rhs2))
    # A + A* is self-adjoint
    checks.append(np.allclose(a + a.T, (a + a.T).T))
    # <x, A* y> for complex-like skew part: norm of Ax vs A* x equal? not equal
    # generally, but singular values coincide
    checks.append(
        np.allclose(np.linalg.svd(a, compute_uv=False), np.linalg.svd(a.T, compute_uv=False))
    )
    # adjoint of projection = itself iff orthogonal
    q = np.linalg.qr(rng.normal(size=(4, 2)))[0]
    p = q @ q.T
    checks.append(np.allclose(p, p.T))
    # oblique projection not self-adjoint
    ob = np.array([[1.0, 0.5], [0.0, 0.0]])
    checks.append(not np.allclose(ob, ob.T))
    return float(sum(checks) / len(checks))


def bench_adjoint_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adjoint_op": _bench_adjoint_op(seed)}
