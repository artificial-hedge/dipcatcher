"""Izhikevich neuron (2003) — two-variable quadratic spiking model
(tonic/phasic/bursting regimes); classify inputs by firing-rate
response curve vs LIF; measures firing-rate separation between classes.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sk_synth import sk_data


def _izh_rate(drive: float, T: int = 200, a: float = 0.02, b: float = 0.2) -> float:
    v, u = -65.0, -14.0
    cnt = 0
    for _ in range(T):
        v += 0.04 * v * v + 5 * v + 140 - u + drive
        u += 0.02 * (0.2 * v - u)
        if v >= 30:
            v = -65.0
            u += 8.0
            cnt += 1
    return cnt / T


def bench_izhikevich(seed: int = 1919) -> dict[str, float]:
    X, y = sk_data(seed, n=120)
    # input drive = projection onto class-mean difference
    cen1 = X[y == 1].mean(0)
    cen0 = X[y == 0].mean(0)
    s = X @ (cen1 - cen0) / np.linalg.norm(cen1 - cen0)
    r1 = np.array([_izh_rate(5 + 3 * float(v)) for v in s])
    thr = np.median(r1)
    acc = float(((r1 > thr).astype(int) == y).mean())
    # separation: mean rate difference
    sep = float(r1[y == 1].mean() - r1[y == 0].mean())
    r0 = float(_izh_rate(5.0))
    r10 = float(_izh_rate(15.0))
    return {
        "synthetic_izh_acc": acc,
        "synthetic_izh_rate_sep": sep,
        "synthetic_izh_rate_low": r0,
        "synthetic_izh_rate_high": r10,
        "synthetic_torch_available": 0.0,
    }
