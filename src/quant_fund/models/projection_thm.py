"""Projection theorem: unique best approximation in a closed subspace (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def ortho_project(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Project x onto span(columns of ON basis q)."""
    return np.asarray(q @ (q.T @ x))


def _bench_projection_thm(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    q, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    x = rng.normal(size=6)
    p = ortho_project(x, q)
    # residual orthogonal to subspace
    checks.append(np.allclose(q.T @ (x - p), 0, atol=1e-12))
    # Pythagoras: ||x||^2 = ||p||^2 + ||x-p||^2
    checks.append(abs(np.dot(x, x) - np.dot(p, p) - np.dot(x - p, x - p)) < 1e-10)
    # best approximation: ||x - p|| <= ||x - y|| for all y in span
    for _ in range(50):
        y = q @ rng.normal(size=2)
        checks.append(np.linalg.norm(x - p) <= np.linalg.norm(x - y) + 1e-12)
    # idempotent + self-adjoint projection
    p_mat = q @ q.T
    checks.append(np.allclose(p_mat @ p_mat, p_mat))
    checks.append(np.allclose(p_mat, p_mat.T))
    return float(sum(checks) / len(checks))


def bench_projection_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_projection_thm": _bench_projection_thm(seed)}
