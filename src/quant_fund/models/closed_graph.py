"""Closed graph theorem in finite dimensions: linear operators are bounded (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def boundedness(a: np.ndarray) -> float:
    """Operator norm bound (always finite in finite dimensions)."""
    return float(np.linalg.svd(a, compute_uv=False)[0])


def _on_graph(a: np.ndarray, x: np.ndarray, y: np.ndarray, tol: float) -> bool:
    """Is (x, y) on the graph of the linear map a? Must discriminate:
    a point off the graph by more than tol is rejected."""
    return bool(np.allclose(np.asarray(a @ x), np.asarray(y), atol=tol, rtol=0.0))


def graph_is_closed(a: np.ndarray, rng: np.random.Generator, n: int = 50) -> bool:
    """Check (x_k, Ax_k) -> (x,y) implies y = Ax on convergent sequences.

    Each trial draws a limit point x and a sequence point x_k = x + eps·v;
    the observed graph limit y_lim = A x_k must lie within the contraction
    bound eps·||A||·||v|| of A x (continuity), AND the same membership
    test must reject a deliberately off-graph limit claim — a check that
    can never reject verifies nothing."""
    bound = max(boundedness(a), 1e-12)
    for _ in range(n):
        x = np.asarray(rng.normal(size=a.shape[1]), dtype=np.float64)
        v = np.asarray(rng.normal(size=a.shape[1]), dtype=np.float64)
        step = 1e-3 * (0.5 + float(rng.random()))
        x_k = x + step * v
        y_lim = a @ x_k
        tol = 2.0 * bound * step * float(np.linalg.norm(v)) + 1e-12
        if not _on_graph(a, x, y_lim, tol):
            return False
        w = np.asarray(rng.normal(size=a.shape[0]), dtype=np.float64)
        if _on_graph(a, x, y_lim + w, tol):
            return False
    return True


def _bench_closed_graph(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    # any matrix defines a closed-graph operator; its norm is finite
    a = rng.normal(size=(3, 3))
    checks.append(graph_is_closed(a, rng))
    checks.append(bool(boundedness(a) < np.inf))
    # ||Ax|| <= ||A|| ||x||
    x = rng.normal(size=3)
    checks.append(bool(float(np.linalg.norm(a @ x)) <= boundedness(a) * np.linalg.norm(x) + 1e-9))
    # discontinuity impossible in finite dims: relative error of ||Ax|| bounded by ||A||
    b = np.array([[100.0, 0.0], [0.0, 0.001]])
    checks.append(bool(boundedness(b) == 100.0))
    # norm continuity: ||Ax_k - Ax|| <= ||A|| ||x_k - x||
    x2 = x + 1e-8 * rng.normal(size=3)
    checks.append(
        bool(
            float(np.linalg.norm(a @ x2 - a @ x)) <= boundedness(a) * np.linalg.norm(x2 - x) + 1e-6
        )
    )
    return float(min(1.0, sum(checks) / len(checks)))


def bench_closed_graph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_closed_graph": _bench_closed_graph(seed)}
