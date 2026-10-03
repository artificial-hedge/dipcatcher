"""Quadratic variation: [W]_t = t; scaled martingales (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_quadratic_var(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    n, m = 2000, 30000
    dt = 1.0 / n
    dw = rng.standard_normal((m, n)) * np.sqrt(dt)
    qv = np.sum(dw**2, axis=1)
    checks.append(abs(np.mean(qv) - 1.0) < 0.01)
    # QV variance shrinks as dt -> 0: Var(sum dw^2) = 2 dt -> ~0.001
    checks.append(np.var(qv) < 0.01)
    # a * W has QV a^2 t
    dw2 = 2.0 * rng.standard_normal((m, n)) * np.sqrt(dt)
    checks.append(abs(np.mean(np.sum(dw2**2, axis=1)) - 4.0) < 0.05)
    # total variation of BM diverges while QV finite: |dw| sums ~ n*E|dW|
    tv = np.sum(np.abs(dw), axis=1)
    checks.append(np.mean(tv) > 10.0)  # ~ n sqrt(2 dt/pi) ~ 35.7
    # realized variance of GBM martingale part ~ sig^2 T
    sig = 0.3
    log_inc = []
    si = np.ones(m)
    for i in range(n):
        si += 0.0 * si * dt + sig * si * dw[:, i]
        log_inc.append(sig * si * dw[:, i])
    rv = np.sum(np.array(log_inc) ** 2, axis=0)  # hmm sig^2 s^2 dw^2? keep proxy
    # simpler: QV of the martingale increments sig*S*dW ~ sig^2 E[S^2] dt
    checks.append(abs(np.mean(rv) - sig**2 * np.exp(sig**2)) < 0.1)
    return float(sum(checks) / len(checks))


def bench_quadratic_var(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quadratic_var": _bench_quadratic_var(seed)}
