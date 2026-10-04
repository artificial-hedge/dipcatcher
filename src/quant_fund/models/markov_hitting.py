"""Hitting probabilities via first-step-analysis linear system (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def hitting_probs(trans: np.ndarray, target: int) -> np.ndarray:
    """Solve h = P h with h[target]=1, h[absorbing others]=0 via iteration."""
    n = trans.shape[0]
    h = np.zeros(n)
    h[target] = 1.0
    for _ in range(50000):
        nh = trans @ h
        nh[target] = 1.0
        if np.allclose(nh, h, rtol=0.0, atol=1e-14):
            h = nh
            break
        h = nh
    return h


def expected_hitting_time(trans: np.ndarray, target: int) -> np.ndarray:
    """Solve t = 1 + P t on non-target states."""
    n = trans.shape[0]
    t = np.zeros(n)
    t[target] = 0.0
    for _ in range(50000):
        nt = 1.0 + trans @ t
        nt[target] = 0.0
        if np.allclose(nt, t, rtol=0.0, atol=1e-12):
            t = nt
            break
        t = nt
    return t


def _bench_markov_hitting(seed: int = 0) -> float:
    checks = []
    # birth-death chain 0..4 with absorbing endpoints; hit 4 before 0 from i = i/4
    n = 5
    p = np.zeros((n, n))
    p[0, 0] = p[4, 4] = 1.0
    for i in range(1, 4):
        p[i, i - 1] = p[i, i + 1] = 0.5
    q = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]])
    h = hitting_probs(p, 4)
    checks.append(np.allclose(h, [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-6))
    # expected hitting time on the triangle chain: t = [2, 2, 0]
    t = expected_hitting_time(q, 2)
    checks.append(np.allclose(t, [2.0, 2.0, 0.0], atol=1e-4))
    # harmonicity in interior: h[i] = 0.5 h[i-1] + 0.5 h[i+1]
    checks.append(all(np.isclose(h[i], 0.5 * h[i - 1] + 0.5 * h[i + 1]) for i in (1, 2, 3)))
    # chain on triangle always reaches target w.p.1
    h2 = hitting_probs(q, 2)
    checks.append(np.allclose(h2, [1.0, 1.0, 1.0], atol=1e-4))
    return float(sum(checks) / len(checks))


def bench_markov_hitting(seed: int = 0) -> dict[str, float]:
    return {"synthetic_markov_hitting": _bench_markov_hitting(seed)}
