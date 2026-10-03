"""Lebesgue integral via level-set (simple-function) approximation (wave 288).

Integral of monotone f on [0,1] by partitioning the RANGE (Lebesgue)
matches the Riemann rectangle integral for continuous f.
"""

import numpy as np

_SEED = 20261231 + 813


def _lebesgue(f, n: int = 4000, m: int = 200) -> float:
    # int f dmu = int_0^inf mu{f > y} dy (for nonnegative f)
    x = np.linspace(0, 1, n)
    y = f(x)
    ys = np.linspace(0, y.max(), m)
    layer = np.array([np.mean(y > t) for t in ys])
    return float(np.trapezoid(layer, ys))


def _riemann(f, n: int = 4000) -> float:
    x = np.linspace(0, 1, n)
    return float(np.trapezoid(f(x), x))


def bench_leb_integral(seed: int = _SEED) -> dict[str, float]:
    for f, want in [
        (lambda x: x * x, 1 / 3),
        (lambda x: np.exp(x), np.e - 1),
        (lambda x: np.sqrt(x), 2 / 3),
    ]:
        val = _lebesgue(f)
        if abs(val - want) / want > 2e-2 or abs(val - _riemann(f)) / want > 2e-2:
            return {"synthetic_leb_integral": 0.0}
    return {"synthetic_leb_integral": 1.0}
