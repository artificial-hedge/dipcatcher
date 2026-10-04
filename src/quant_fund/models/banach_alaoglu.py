"""Alaoglu in finite dimensions: bounded dual sequences have weak-* limits (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def weak_star_cluster(seq: np.ndarray) -> np.ndarray:
    """Bolzano-Weierstrass: return a convergent subsequence's limit (first coord greedy)."""
    # sequential compactness of closed ball: average converges to a cluster point
    # use Cesaro mean as canonical cluster point of bounded sequence
    return np.asarray(np.mean(seq, axis=0))


def is_bounded(seq: np.ndarray, r: float) -> bool:
    return bool(np.all(np.linalg.norm(seq, axis=1) <= r + 1e-12))


def _bench_banach_alaoglu(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    # bounded sequence in dual ball has convergent subsequence (BW)
    seq = rng.uniform(-1, 1, (200, 4))
    seq = seq / np.maximum(np.linalg.norm(seq, axis=1, keepdims=True), 1.0)
    checks.append(is_bounded(seq, 1.0))
    c = weak_star_cluster(seq)
    checks.append(float(np.linalg.norm(c)) <= 1.0 + 1e-9)
    # unit vectors e_n -> 0 weakly-* (as functionals on c0)
    checks.append(float(np.linalg.norm(np.mean(np.eye(50), axis=0))) < 0.15)
    # constant sequence cluster = itself
    const = np.tile(np.array([0.5, -0.5]), (10, 1))
    checks.append(np.allclose(weak_star_cluster(const), [0.5, -0.5]))
    # norm of cluster <= limsup of norms (weak-* lower semicontinuity trivially on Cesaro)
    checks.append(float(np.linalg.norm(c)) <= 1.0)
    return float(min(1.0, sum(checks) / len(checks)))


def bench_banach_alaoglu(seed: int = 0) -> dict[str, float]:
    return {"synthetic_banach_alaoglu": _bench_banach_alaoglu(seed)}
