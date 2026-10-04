"""Gardner's relation rho = a * V^b: empirical density-velocity rule.

Log-log least squares recovers (a, b); classic brine-saturated sediment
values a=0.31, b=0.25 (V in m/s, rho in g/cc).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 936


def gardner_rho(v: np.ndarray, a: float = 0.31, b: float = 0.25) -> np.ndarray:
    return a * np.asarray(v, dtype=np.float64) ** b


def fit_gardner(v: np.ndarray, rho: np.ndarray) -> tuple[float, float]:
    x = np.log(np.asarray(v, dtype=np.float64))
    y = np.log(np.asarray(rho, dtype=np.float64))
    A = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.exp(coef[0])), float(coef[1])


def bench_gardner_relation(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a_true, b_true = 0.31, 0.25
    v = rng.uniform(1500.0, 4500.0, 400)
    rho = gardner_rho(v, a_true, b_true) * np.exp(rng.normal(0.0, 0.01, v.shape))
    a_hat, b_hat = fit_gardner(v, rho)
    pred = gardner_rho(v, a_hat, b_hat)
    rel = np.abs(pred - rho) / rho
    resid = np.log(rho) - np.log(pred)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((np.log(rho) - np.log(rho).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    checks = [
        abs(a_hat - a_true) < 0.05,
        abs(b_hat - b_true) < 0.03,
        r2 > 0.95,
        float(rel.mean()) < 0.05,
        float(rel.max()) < 0.15,
    ]
    return {"synthetic_gardner_relation": float(np.mean(checks))}
