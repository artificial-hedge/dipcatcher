"""Open mapping principle in finite dimensions: surjective linear maps are open (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def image_of_ball(a: np.ndarray, rng: np.random.Generator, n: int = 4000) -> np.ndarray:
    """Images of random points in the unit ball under A."""
    x = rng.normal(size=(n, a.shape[1]))
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0)
    return np.asarray(x @ a.T)


def min_image_norm(a: np.ndarray) -> float:
    """Smallest ||Ax|| over ||x||=1 = smallest singular value."""
    return float(np.linalg.svd(a, compute_uv=False)[-1])


def _bench_open_mapping(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    # surjective A (2x2 invertible): image of unit ball contains ball of radius sigma_min
    a = np.array([[2.0, 0.5], [0.0, 1.0]])
    smin = min_image_norm(a)
    checks.append(abs(smin - np.linalg.svd(a, compute_uv=False)[-1]) < 1e-9)
    checks.append(smin > 0.5)
    # image ball covers a ball of radius ~smin: check directions reachable with norm >= smin*0.9
    ys = image_of_ball(a, rng)
    for theta in np.linspace(0, 2 * np.pi, 12):
        d = np.array([np.cos(theta), np.sin(theta)])
        best = float(np.max(ys @ d))
        checks.append(best >= smin * 0.9)
    # non-surjective: rank-1 map collapses one direction (min image norm ~ 0)
    b = np.array([[1.0, 1.0], [1.0, 1.0]])
    checks.append(min_image_norm(b) < 1e-12)
    return float(min(1.0, sum(checks) / len(checks)))


def bench_open_mapping(seed: int = 0) -> dict[str, float]:
    return {"synthetic_open_mapping": _bench_open_mapping(seed)}
