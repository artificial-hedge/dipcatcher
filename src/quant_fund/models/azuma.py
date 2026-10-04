"""Azuma-Hoeffding: P(M_n - M_0 >= t) <= exp(-t^2 / (2 sum c_k^2)) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_azuma(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    n = 100
    steps = rng.choice([-1.0, 1.0], size=(40000, n))
    s = steps.sum(axis=1)
    # Azuma bound with c_k = 1: P(S_n >= t) <= exp(-t^2 / 2n)
    for t in (20.0, 30.0, 40.0):
        emp = float(np.mean(s >= t))
        bound = float(np.exp(-(t**2) / (2 * n)))
        checks.append(emp <= bound + 1e-4)
    # bound holds loosely but correctly ordered: bound decreases with t
    bs = [float(np.exp(-(t**2) / (2 * n))) for t in (10.0, 20.0, 30.0)]
    checks.append(bs[0] > bs[1] > bs[2])
    # two-sided version doubles the bound
    emp2 = float(np.mean(np.abs(s) >= 30.0))
    checks.append(emp2 <= 2 * float(np.exp(-(30.0**2) / (2 * n))) + 1e-4)
    # sub-Gaussian MGF: E[exp(lambda S_n)] <= exp(n lambda^2 / 2) at small lambda
    lam = 0.1
    mgf = float(np.mean(np.exp(lam * s)))
    checks.append(mgf <= np.exp(n * lam**2 / 2) * 1.05)
    return float(sum(checks) / len(checks))


def bench_azuma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_azuma": _bench_azuma(seed)}
