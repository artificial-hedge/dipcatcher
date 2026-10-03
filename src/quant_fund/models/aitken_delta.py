"""Aitken delta-squared acceleration for linearly convergent sequences (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def aitken(seq: np.ndarray) -> np.ndarray:
    """Delta^2 transform: s_n - (d s_n)^2 / d^2 s_n."""
    d1 = seq[1:] - seq[:-1]
    d2 = d1[1:] - d1[:-1]
    denom = np.where(np.abs(d2) < 1e-300, np.inf, d2)
    out = seq[:-2] - d1[:-1] ** 2 / denom
    out = np.where(np.isfinite(out), out, seq[:-2])
    return np.asarray(out)


def _bench_aitken_delta(seed: int = 0) -> float:
    checks = []
    # fixed point iteration x = cos(x): converges linearly to Dottie number
    x = 0.5
    seq = [x]
    for _ in range(40):
        x = float(np.cos(x))
        seq.append(x)
    s = np.array(seq)
    acc = aitken(s)
    truth = 0.7390851332151607
    checks.append(abs(acc[-1] - truth) < 1e-8)
    # accelerated beats raw tail
    checks.append(abs(acc[-1] - truth) < abs(s[-1] - truth))
    # geometric series ratio 1/2: partial sums -> 1 exactly after transform
    geo = np.cumsum(0.5 ** np.arange(1, 30))
    ga = aitken(geo)
    checks.append(abs(ga[-1] - 1.0) < 1e-10)
    # alternating harmonic -> ln 2: transform improves partial sum
    alt = np.cumsum([(-1.0) ** (k + 1) / k for k in range(1, 60)])
    aa = aitken(alt)
    checks.append(abs(aa[-1] - np.log(2.0)) < abs(alt[-1] - np.log(2.0)))
    # constant sequence unaffected
    checks.append(np.allclose(aitken(np.full(10, 3.5)), 3.5))
    return float(sum(checks) / len(checks))


def bench_aitken_delta(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aitken_delta": _bench_aitken_delta(seed)}
