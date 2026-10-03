"""Uniform boundedness (Banach-Steinhaus) in finite dimensions (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def pointwise_bound(ops: list[np.ndarray], x: np.ndarray) -> float:
    return float(max(float(np.linalg.norm(t @ x)) for t in ops))


def operator_norm_bound(ops: list[np.ndarray]) -> float:
    return float(max(float(np.linalg.svd(t, compute_uv=False)[0]) for t in ops))


def _bench_uniform_bounded(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    # family of rotations scaled by <= 2: pointwise bounded -> uniformly bounded by 2
    ops = [
        2.0 * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        for t in np.linspace(0, 1, 20)
    ]
    checks.append(bool(operator_norm_bound(ops) <= 2.0 + 1e-9))
    for _ in range(10):
        x = rng.normal(size=2)
        checks.append(bool(pointwise_bound(ops, x) <= 2.0 * np.linalg.norm(x) + 1e-9))
    # unbounded family: ops = n*I has pointwise bound inf at every nonzero x
    ops2 = [float(n) * np.eye(2) for n in range(1, 100)]
    checks.append(bool(operator_norm_bound(ops2) >= 99.0))
    checks.append(bool(pointwise_bound(ops2, np.array([1.0, 0.0])) >= 99.0))
    # resonance: supremum of ||T x|| over x equals operator norm
    t = np.diag([3.0, 1.0])
    checks.append(bool(abs(operator_norm_bound([t]) - 3.0) < 1e-9))
    return float(min(1.0, sum(checks) / len(checks)))


def bench_uniform_bounded(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uniform_bounded": _bench_uniform_bounded(seed)}
