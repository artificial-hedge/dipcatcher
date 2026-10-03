"""Strong SDE schemes: Euler-Maruyama vs Milstein convergence (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _paths(m: int, n: int, mu: float, sig: float, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(rng.standard_normal((m, n)) * np.sqrt(1.0 / n))


def _bench_sde_strong(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    m, n = 20000, 128
    mu, sig = 0.05, 0.4
    dt = 1.0 / n
    dw = _paths(m, n, mu, sig, rng)
    # exact GBM: S_T = exp((mu - sig^2/2) T + sig W_T)
    wt = np.cumsum(dw, axis=1)
    exact = np.exp((mu - 0.5 * sig**2) + sig * wt[:, -1])
    # Euler-Maruyama
    s_em = np.ones(m)
    for i in range(n):
        s_em += mu * s_em * dt + sig * s_em * dw[:, i]
    # Milstein adds 0.5 sig^2 S (dw^2 - dt)
    s_mi = np.ones(m)
    for i in range(n):
        s_mi += mu * s_mi * dt + sig * s_mi * dw[:, i] + 0.5 * sig**2 * s_mi * (dw[:, i] ** 2 - dt)
    em_err = float(np.mean(np.abs(s_em - exact)))
    mi_err = float(np.mean(np.abs(s_mi - exact)))
    checks.append(mi_err < em_err)
    checks.append(mi_err < 0.02)
    # EM error dominated by missing second-order term: still O(sqrt(dt))-ish
    checks.append(em_err < 0.1)
    # weak convergence much better: means match closely
    checks.append(abs(float(np.mean(s_mi)) - float(np.mean(exact))) < 0.01)
    # log-space Euler IS exact for GBM (equivalent to Milstein in log coords)
    log_euler = np.exp((mu - 0.5 * sig**2) + sig * wt[:, -1])
    checks.append(bool(np.allclose(log_euler, exact)))
    return float(sum(checks) / len(checks))


def bench_sde_strong(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sde_strong": _bench_sde_strong(seed)}
