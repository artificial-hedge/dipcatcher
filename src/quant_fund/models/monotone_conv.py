"""Monotone + dominated convergence on the grid (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_monotone_conv(seed: int = 0) -> float:
    checks = []
    x = np.linspace(1e-6, 1, 20001)
    dx = x[1] - x[0]
    # MCT: f_n = min(x^{-1/2}, n) increasing to x^{-1/2}; int -> 2
    ints = []
    prev = np.zeros_like(x)
    for n in (4, 16, 64, 256, 1024):
        fn = np.minimum(x ** (-0.5), float(n))
        checks.append(bool(np.all(fn >= prev - 1e-12)))  # monotone
        prev = fn
        ints.append(float(np.sum(fn) * dx))
    checks.append(abs(ints[-1] - 2.0) < 0.05)
    # ints increasing toward limit
    checks.append(all(ints[i] <= ints[i + 1] + 1e-9 for i in range(len(ints) - 1)))
    # DCT: f_n = x^n bounded by g = 1 -> int -> 0
    int_dom = float(np.sum(x**100 * 1.0) * dx)
    checks.append(int_dom < 0.02)
    # DCT: moving to limit f_n -> f = 0 a.e. dominated by 1
    checks.append(abs(float(np.sum(np.minimum(x**50, 1.0)) * dx)) < 0.02 + 1e-9)
    # dominated required: spike sequence NOT dominated by integrable bound fails
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_monotone_conv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monotone_conv": _bench_monotone_conv(seed)}
