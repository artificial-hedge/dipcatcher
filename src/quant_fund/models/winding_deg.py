"""Degree of a circle map S^1 -> S^1 via discrete winding (wave 280) (SYNTHETIC).

Sample the lifted map on a fine grid; the degree equals the signed
wrap-count sum of angle increments over one period.
"""

from collections.abc import Callable

import numpy as np

_SEED = 20261231 + 769


def degree(f: Callable[[float], float], n: int = 2000) -> int:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vals = np.unwrap(np.array([f(t) for t in th]))
    return int(round(float(vals[-1] - vals[0] + (f(2 * np.pi) - f(0)) / (2 * np.pi)) / (2 * np.pi)))


def _wrap_deg(f: Callable[[float], float], n: int = 512) -> int:
    th = np.linspace(0, 2 * np.pi, n + 1)
    d = np.diff(np.array([f(t) for t in th]))
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap increments to [-pi, pi)
    return int(round(float(d.sum()) / (2 * np.pi)))


def bench_winding_deg(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    trials = 0

    def _mk(k: float, a: float, b: float) -> Callable[[float], float]:
        def f(t: float) -> float:
            return float(k * t + a + b * np.sin(3 * t))

        return f

    for k in (-2, -1, 0, 1, 2, 3):
        a = rng.uniform(0, 2 * np.pi)
        b = rng.uniform(0.05, 0.3)
        got = _wrap_deg(_mk(k, a, b))
        ok += int(got == k)
        trials += 1
    return {"synthetic_winding_deg": float(ok == trials)}
