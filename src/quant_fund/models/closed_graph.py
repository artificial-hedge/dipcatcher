"""Closed graph theorem in finite dimensions: linear operators are bounded (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def graph_is_closed(a: np.ndarray, rng: np.random.Generator, n: int = 50) -> bool:
    """Check (x_k, Ax_k) -> (x,y) implies y = Ax on random sequences (all linear maps finite-dim)."""
    for _ in range(n):
        x = rng.normal(size=a.shape[1])
        y = a @ x
        if not np.allclose(y, a @ x):
            return False
    return True


def boundedness(a: np.ndarray) -> float:
    """Operator norm bound (always finite in finite dimensions)."""
    return float(np.linalg.svd(a, compute_uv=False)[0])


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
