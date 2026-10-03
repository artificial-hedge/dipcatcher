"""Girsanov change of measure for shifted Brownian drift (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_girsanov(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    n, m = 200, 80000
    dt = 1.0 / n
    theta = 0.5  # shift: under Q, W^Q = W + theta t is BM
    dw = rng.standard_normal((m, n)) * np.sqrt(dt)
    w = np.cumsum(dw, axis=1)
    wt = w[:, -1]
    # Radon-Nikodym derivative dQ/dP = exp(-theta W_T - theta^2 T/2)
    # for W^Q = W + theta t Brownian under Q
    lrn = -theta * wt - 0.5 * theta**2
    rn = np.exp(lrn)
    checks.append(abs(np.mean(rn) - 1.0) < 0.02)
    # E_Q[W^Q_T] = E_P[(W_T + theta) * RN] should be ~0
    checks.append(abs(np.mean((wt + theta) * rn)) < 0.02)
    # E_P[W_T + theta] = theta (drift under P)
    checks.append(abs(np.mean(wt + theta) - theta) < 0.02)
    # E_Q[(W^Q_T)^2] = T = 1
    checks.append(abs(np.mean((wt + theta) ** 2 * rn) - 1.0) < 0.05)
    # log-likelihood variance = theta^2 T
    checks.append(abs(np.var(lrn) - theta**2) < 0.05)
    return float(sum(checks) / len(checks))


def bench_girsanov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_girsanov": _bench_girsanov(seed)}
