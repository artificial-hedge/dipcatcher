"""Romberg integration: Richardson extrapolation of trapezoid rule (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def romberg(f, a: float, b: float, levels: int = 6) -> np.ndarray:
    r = np.zeros((levels, levels))
    for i in range(levels):
        n = 2**i
        xs = np.linspace(a, b, n + 1)
        r[i, 0] = (b - a) / n * (0.5 * f(xs[0]) + np.sum(f(xs[1:-1])) + 0.5 * f(xs[-1]))
        for j in range(1, i + 1):
            r[i, j] = r[i, j - 1] + (r[i, j - 1] - r[i - 1, j - 1]) / (4**j - 1)
    return r


def _bench_romberg(seed: int = 0) -> float:
    checks = []
    # int_0^pi sin x dx = 2
    r = romberg(np.sin, 0.0, np.pi, 7)
    checks.append(abs(r[-1, -1] - 2.0) < 1e-12)
    # column j+1 improves over column j
    checks.append(abs(r[-1, 3] - 2.0) < abs(r[-1, 0] - 2.0))
    # trapezoid error halves-quarters: E_{2n} ~ E_n / 4
    e1 = abs(r[4, 0] - 2.0)
    e2 = abs(r[5, 0] - 2.0)
    checks.append(0.15 < e2 / e1 < 0.35)
    # int_0^1 x^4 dx = 0.2 exact for cubic-extrapolated? R[2,2] uses Simpson^2
    r2 = romberg(lambda x: x**4, 0.0, 1.0, 5)
    checks.append(abs(r2[-1, -1] - 0.2) < 1e-12)
    # int_0^1 e^x = e - 1
    r3 = romberg(np.exp, 0.0, 1.0, 6)
    checks.append(abs(r3[-1, -1] - (np.e - 1)) < 1e-11)
    return float(sum(checks) / len(checks))


def bench_romberg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_romberg": _bench_romberg(seed)}
