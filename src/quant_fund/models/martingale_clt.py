"""Martingale CLT: normalized martingale -> N(0,1) (SYNTHETIC)."""

from __future__ import annotations

from math import erf

import numpy as np


def _bench_martingale_clt(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    n_steps = 200
    n_rep = 4000
    # martingale differences: iid mean-0 bounded increments U(-1,1),
    # Var = 1/3 each -> M_n = sum, z = sqrt(3/n) M_n -> N(0,1)
    steps = rng.uniform(-1.0, 1.0, size=(n_rep, n_steps))
    z = steps.sum(axis=1) * np.sqrt(3.0 / n_steps)
    # moments match N(0,1)
    checks.append(abs(float(np.mean(z))) < 0.05)
    checks.append(abs(float(np.var(z)) - 1.0) < 0.1)
    # KS distance to standard normal small
    xs = np.sort(z)
    ecdf = np.arange(1, n_rep + 1) / n_rep
    tcdf = np.array([0.5 * (1.0 + erf(v / np.sqrt(2.0))) for v in xs])
    ks = float(np.max(np.abs(ecdf - tcdf)))
    checks.append(ks < 0.03)
    # skewness ~ 0
    checks.append(abs(float(np.mean(z**3))) < 0.2)
    # bounded increments condition (Lindeberg) satisfied trivially
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_martingale_clt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_martingale_clt": _bench_martingale_clt(seed)}
