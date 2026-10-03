"""Tanaka formula: |B_t| = int sgn(B) dB + L_t^0 (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_tanaka(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = 20_000
    dt = 1.0 / n
    db = rng.standard_normal(n) * np.sqrt(dt)
    b = np.concatenate([[0.0], np.cumsum(db)])
    checks = []
    # local time estimate: L_t^0 ≈ lim (1/2eps) int 1{|B_s|<eps} ds
    eps = 0.05
    occupation = float(np.mean(np.abs(b) < eps))
    local_time = occupation / (2 * eps)
    # stochastic integral int_0^t sgn(B_s) dB_s
    sgn = np.sign(b[:-1])
    stoch_int = float(np.sum(sgn * db))
    # Tanaka residual: |B_t| - int sgn(B)dB should equal L_t^0 ≈ local_time
    residual = abs(b[-1]) - stoch_int
    checks.append(abs(residual - local_time) < 0.35)
    # |B_t| is nonnegative
    checks.append(abs(b[-1]) >= 0)
    # local time is positive for a path starting at 0
    checks.append(local_time > 0)
    # sgn integrand bounded -> finite integral
    checks.append(abs(stoch_int) < 10.0)
    # occupation fraction in |B| < eps is small but positive
    checks.append(0.0 < occupation < 0.2)
    return float(sum(checks) / len(checks))


def bench_tanaka(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tanaka": _bench_tanaka(seed)}
