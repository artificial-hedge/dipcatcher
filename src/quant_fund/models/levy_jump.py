"""Merton jump-diffusion: dS = μS dt + σS dW + S dJ, J = Σ(e^{Y_i}-1) (SYNTHETIC)
with Y ~ N(μj, σj), Poisson(λ). Empirical excess kurtosis/skew vs
pure GBM — jump signature detection.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def _merton(
    s0: float,
    mu: float,
    sig: float,
    lam: float,
    mj: float,
    sj: float,
    T: float,
    steps: int,
    rng,
) -> np.ndarray:
    dt = T / steps
    log_s = np.log(s0)
    path = [s0]
    for _ in range(steps):
        nj = rng.poisson(lam * dt)
        j = sum(rng.normal(mj, sj) for _ in range(nj))
        log_s += (
            (mu - 0.5 * sig**2 - lam * (np.exp(mj + sj**2 / 2) - 1)) * dt
            + sig * np.sqrt(dt) * rng.standard_normal()
            + j
        )
        path.append(np.exp(log_s))
    return np.asarray(path)


def bench_levy_jump(seed: int = 2951, paths: int = 400) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rets_j: list[float] = []
    rets_g: list[float] = []
    for _ in range(paths):
        p = _merton(100, 0.05, 0.15, 3.0, -0.02, 0.05, 0.25, 60, rng)
        rets_j.extend(np.diff(np.log(p)))
        g = 100 * np.exp(
            np.cumsum(
                (0.05 - 0.5 * 0.15**2) * (0.25 / 60)
                + 0.15 * np.sqrt(0.25 / 60) * rng.standard_normal(60)
            )
        )
        g = np.concatenate([[100.0], g])
        rets_g.extend(np.diff(np.log(g)))
    k_j = float(kurtosis(np.asarray(rets_j)))
    k_g = float(kurtosis(np.asarray(rets_g)))
    s_j = float(skew(np.asarray(rets_j)))
    return {
        "synthetic_merton_kurt": k_j,
        "synthetic_gbm_kurt": k_g,
        "synthetic_merton_kurt_excess": k_j - k_g,
        "synthetic_merton_skew": s_j,
        "synthetic_torch_available": 0.0,
    }
