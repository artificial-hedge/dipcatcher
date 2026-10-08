"""Doob decomposition: X = M + A with A predictable, increasing (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def srw_path(n: int, rng: np.random.Generator) -> np.ndarray:
    steps = rng.choice([-1.0, 1.0], size=n)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _bench_doob_decomp(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # X_n = S_n^2 is a submartingale; compensator A_n = n, martingale M = S^2 - n
    n = 50
    a = np.arange(n + 1, dtype=float)  # compensator of S^2 is A_n = n
    # A is predictable (deterministic) and increasing
    checks.append(bool(np.all(np.diff(a) >= 0)))
    # M is a martingale: E[M_{k+1} - M_k | F_k] = 0 empirically
    diffs = []
    for _ in range(20000):
        s2 = srw_path(n, rng)
        m2 = s2**2 - np.arange(n + 1)
        diffs.append(np.diff(m2))
    darr = np.array(diffs)
    # pooled increment mean ~ 0 (per-position se too noisy at tail positions)
    checks.append(abs(float(np.mean(darr))) < 0.02)
    # per-position martingale property: mean of ΔM at a FIXED position
    # k=10 must also be ~0 (se = sd/sqrt(20000) ≈ 0.014)
    checks.append(abs(float(darr[:, 10].mean())) < 0.05)
    # empirical A estimate = cumulative predictable increment of X:
    # E[X_{k+1} - X_k | F_k] = 1 for X = S^2 -> A_n = n
    incr = np.array([np.diff(srw_path(n, rng) ** 2) for _ in range(20000)])
    checks.append(abs(float(np.mean(incr)) - 1.0) < 0.02)
    # supermartingale X = -S^2 has compensator drift -1 per step:
    # E[-(S_{k+1}^2 - S_k^2) | F_k] = -1, i.e. M' = -S^2 + n is a martingale
    decr = np.array([-np.diff(srw_path(n, rng) ** 2) for _ in range(8000)])
    checks.append(abs(float(np.mean(decr)) + 1.0) < 0.03)
    return float(sum(checks) / len(checks))


def bench_doob_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doob_decomp": _bench_doob_decomp(seed)}
