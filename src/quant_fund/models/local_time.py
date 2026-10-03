"""Brownian local time and occupation-time density (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_local_time(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    n, m = 4000, 20000
    dt = 1.0 / n
    dw = rng.standard_normal((m, n)) * np.sqrt(dt)
    w = np.concatenate([np.zeros((m, 1)), np.cumsum(dw, axis=1)], axis=1)
    # occupation time of |x| < eps ~ 2 eps * L(0) / dt-scaled: L^0 ~ time fraction
    eps = 0.05
    occ = np.mean(np.abs(w) < eps, axis=1)  # fraction of steps near 0
    # theoretical: E[occupation fraction in (-eps,eps)] ~ 2 eps / sqrt(pi T/2)
    # = 2 eps * E|N(0,T)| / T /? For BM local time: occupation density at 0 ~ L_T(0)
    # E[L_T(0)] = E|W_T| = sqrt(2T/pi); occupation frac ~ eps * L * 2 / T
    pred = 2 * eps * np.sqrt(2 / np.pi)
    checks.append(abs(np.mean(occ) - pred) < 0.015)
    # |W_T| is distributed like local time at 0 (Levy's theorem): same moments
    checks.append(abs(np.mean(np.abs(w[:, -1])) - np.sqrt(2 / np.pi)) < 0.02)
    # occupation shrinks with eps -> linear in eps
    occ2 = np.mean(np.abs(w) < 2 * eps, axis=1)
    ratio = np.mean(occ2) / np.mean(occ)
    checks.append(1.6 < ratio < 2.4)
    # arcsine law: time above 0 has arcsine distribution; P(frac<0.2) ~ asin
    frac_pos = np.mean(w > 0, axis=1)
    p20 = np.mean(frac_pos < 0.2)
    asin = (2 / np.pi) * np.arcsin(np.sqrt(0.2))
    checks.append(abs(p20 - asin) < 0.02)
    return float(sum(checks) / len(checks))


def bench_local_time(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_time": _bench_local_time(seed)}
