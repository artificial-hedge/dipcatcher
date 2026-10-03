"""Birkhoff ergodic theorem on a finite ergodic Markov chain (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_ergodic_thm(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # chain on {0,1,2}: circulant P with p_ij = 0.7 stay / 0.3 forward
    p = np.array([[0.7, 0.3, 0.0], [0.0, 0.7, 0.3], [0.3, 0.0, 0.7]])
    # stationary distribution (uniform: doubly stochastic columns sum to 1)
    checks.append(np.allclose(p.sum(axis=0), 1.0))
    pi = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    checks.append(np.allclose(pi @ p, pi))
    # simulate long path; time average of f(x) = x -> E_pi f = 1.0
    n = 200000
    x = 0
    tot = 0.0
    count = np.zeros(3)
    for _ in range(n):
        x = int(rng.choice(3, p=p[x]))
        tot += x
        count[x] += 1
    checks.append(abs(tot / n - 1.0) < 0.02)
    # occupation fractions -> pi
    checks.append(np.allclose(count / n, pi, atol=0.02))
    # time avg of indicator_{x=2} -> 1/3
    checks.append(abs(count[2] / n - 1.0 / 3) < 0.02)
    return float(sum(checks) / len(checks))


def bench_ergodic_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ergodic_thm": _bench_ergodic_thm(seed)}
