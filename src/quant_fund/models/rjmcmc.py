"""RJMCMC (Green 1995): reversible-jump moves between a 1-component and (SYNTHETIC)
2-component Gaussian-mixture fit to bimodal data; posterior model
probabilities should favor the truth (k=2 on bimodal, k=1 on unimodal).
"""

from __future__ import annotations

import numpy as np


def _ll1(y: np.ndarray, mu: float, sd: float) -> float:
    return float((-0.5 * ((y - mu) / sd) ** 2 - np.log(sd)).sum())


def _ll2(y: np.ndarray, p: float, m1: float, m2: float, sd: float) -> float:
    d1 = np.exp(-0.5 * ((y - m1) / sd) ** 2) / sd
    d2 = np.exp(-0.5 * ((y - m2) / sd) ** 2) / sd
    return float(np.log(p * d1 + (1 - p) * d2 + 1e-12).sum())


def bench_rjmcmc(seed: int = 2977, steps: int = 1200) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for tag, y in {
        "bimodal": np.concatenate([rng.normal(-2, 0.6, 200), rng.normal(2, 0.6, 200)]),
        "unimodal": rng.normal(0, 0.8, 400),
    }.items():
        k = 1
        m1, m2, p, sd = 0.0, 0.0, 0.5, 0.6
        k2 = 0
        for i in range(steps):
            # within-model MH jitter
            sd = max(0.2, sd + 0.05 * rng.standard_normal())
            if k == 1:
                m1 += 0.1 * rng.standard_normal()
                # birth: propose 2-comp
                pm1 = m1 - 1.5 + rng.standard_normal()
                pm2 = m1 + 1.5 + rng.standard_normal()
                pp = rng.beta(2, 2)
                logr = _ll2(y, pp, pm1, pm2, sd) - _ll1(y, m1, sd) + np.log(0.5)
                if np.log(rng.uniform()) < logr:
                    k, m1, m2, p = 2, pm1, pm2, pp
            else:
                m1 += 0.1 * rng.standard_normal()
                m2 += 0.1 * rng.standard_normal()
                p = float(np.clip(p + 0.05 * rng.standard_normal(), 0.05, 0.95))
                # death: propose 1-comp
                pm = p * m1 + (1 - p) * m2
                logr = _ll1(y, pm, sd) - _ll2(y, p, m1, m2, sd) + np.log(2)
                if np.log(rng.uniform()) < logr:
                    k, m1 = 1, pm
            if i > steps // 3 and k == 2:
                k2 += 1
        out[tag] = k2 / (steps - steps // 3)
    return {
        "synthetic_rj_p2_bimodal": float(out["bimodal"]),
        "synthetic_rj_p2_unimodal": float(out["unimodal"]),
        "synthetic_rj_sep": float(out["bimodal"] - out["unimodal"]),
        "synthetic_torch_available": 0.0,
    }
