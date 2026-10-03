"""Martingale property checks: SRW, square minus n, product martingale (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def srw_path(steps: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(rng.choice([-1, 1], steps))])


def cond_expect_martingale(paths: np.ndarray, t: int) -> float:
    """mean of X_{t+1} - X_t across paths (should be ~0 for martingale)."""
    return float(np.mean(paths[:, t + 1] - paths[:, t]))


def _bench_martingale_check(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    n_path, n_step = 40_000, 20
    paths = np.stack([srw_path(n_step, rng) for _ in range(n_path)])
    # E[X_{t+1} - X_t | F_t] = 0 empirically
    checks.append(abs(cond_expect_martingale(paths, 10)) < 0.02)
    # S_n^2 - n is a martingale: E[S_{t+1}^2 - (t+1) | ...] - (S_t^2 - t) has mean 0
    sq = paths**2 - np.arange(n_step + 1)
    checks.append(abs(cond_expect_martingale(sq, 10)) < 0.05)
    # empirical variance of SRW at time t = t
    checks.append(abs(float(np.var(paths[:, 15])) - 15.0) / 15.0 < 0.1)
    # double-or-nothing martingale product: prod of (1+X_t) with E X = 0 -> e.g. 2*coin-1
    prod = np.prod(rng.choice([0.5, 1.5], (n_path, n_step)), axis=1)
    checks.append(abs(float(np.mean(prod)) - 1.0) < 0.08)
    # non-martingale: drift walk has E[X_t] = t*p
    drift_paths = np.cumsum(rng.choice([0, 1], (n_path, n_step)), axis=1)
    checks.append(abs(float(np.mean(drift_paths[:, 9])) - 5.0) < 0.05)
    return float(sum(checks) / len(checks))


def bench_martingale_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_martingale_check": _bench_martingale_check(seed)}
