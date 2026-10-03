"""Finite Markov chains: Chapman-Kolmogorov + stationary distribution (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def is_stochastic(p: np.ndarray) -> bool:
    return bool(np.all(p >= -1e-12) and np.allclose(p.sum(axis=1), 1.0))


def matrix_power(p: np.ndarray, n: int) -> np.ndarray:
    out = np.eye(p.shape[0])
    for _ in range(n):
        out = out @ p
    return np.asarray(out)


def stationary(p: np.ndarray, iters: int = 500) -> np.ndarray:
    v = np.ones(p.shape[0]) / p.shape[0]
    for _ in range(iters):
        v = v @ p
        v = v / v.sum()
    return np.asarray(v)


def _bench_markov_chain(seed: int = 0) -> float:
    checks = []
    p = np.array([[0.5, 0.5], [0.2, 0.8]])
    checks.append(is_stochastic(p))
    # Chapman-Kolmogorov: P^2 = P P
    checks.append(np.allclose(matrix_power(p, 2), p @ p))
    # stationary dist of [[0.5,0.5],[0.2,0.8]]: pi solves pi1*0.5 = pi2*0.2 -> pi = (2/7,5/7)
    pi = stationary(p)
    checks.append(np.allclose(pi, [2.0 / 7, 5.0 / 7], atol=1e-3))
    checks.append(np.allclose(pi @ p, pi))
    # doubly stochastic -> uniform stationary
    q = np.array([[0.5, 0.5], [0.5, 0.5]])
    checks.append(np.allclose(stationary(q), [0.5, 0.5], atol=1e-6))
    # n-step transition probabilities valid
    checks.append(is_stochastic(matrix_power(p, 10)))
    return float(sum(checks) / len(checks))


def bench_markov_chain(seed: int = 0) -> dict[str, float]:
    return {"synthetic_markov_chain": _bench_markov_chain(seed)}
