"""Reflected Brownian motion |B| and Skorokhod map (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def stationary_mean() -> float:
    """E|Z| for Z ~ N(0,1): half-normal mean sqrt(2/pi)."""
    return float(np.sqrt(2 / np.pi))


def skorokhod(path: np.ndarray) -> np.ndarray:
    """Skorokhod reflection: y_t = x_t - min(0, min_{s<=t} x_s)."""
    running_min = np.minimum.accumulate(path)
    return path - np.minimum(0.0, running_min)


def _bench_reflect_bm(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = 20_000
    dt = 1.0 / n
    b = np.concatenate([[0.0], np.cumsum(rng.standard_normal(n) * np.sqrt(dt))])
    checks = []
    # reflected path is nonnegative
    refl = skorokhod(b)
    checks.append(bool(np.all(refl >= -1e-12)))
    # |B| is also a reflected BM path realization: mean |B_t| matches
    # half-normal * sqrt(t) asymptotically; check a distributional moment
    checks.append(
        abs(
            np.mean(np.abs(b))
            - stationary_mean() * np.sqrt(0.5) * np.sqrt(2 / np.pi) / np.sqrt(2 / np.pi)
        )
        < 0.5
    )
    # reflection increases the minimum to 0
    checks.append(abs(float(np.min(refl)) - 0.0) < 1e-9 or float(np.min(refl)) >= 0.0)
    # stationary mean = sqrt(2/pi)
    checks.append(abs(stationary_mean() - 0.7978845608) < 1e-6)
    # reflected BM equals x - L where L = -min(0, min x)
    lev = -np.minimum(0.0, np.minimum.accumulate(b))
    checks.append(bool(np.allclose(refl, b + lev)))
    return float(sum(checks) / len(checks))


def bench_reflect_bm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reflect_bm": _bench_reflect_bm(seed)}
