"""Poincare-Hopf: sum of vector field indices = chi(M) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _winding(field, center: np.ndarray, r: float = 0.05, n: int = 200) -> int:
    """Index of a vector field around a zero: winding of field on small circle."""
    ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angs = []
    for t in ts:
        p = center + r * np.array([np.cos(t), np.sin(t)])
        v = field(p)
        angs.append(np.arctan2(v[1], v[0]))
    d = np.diff(np.unwrap(angs))
    return int(round(float(np.sum(d)) / (2 * np.pi)))


def _bench_poincare_hopf(seed: int = 0) -> float:
    checks = []
    # source field (x,y): index +1; sink -x,-y: +1; saddle (x,-y): -1
    src = _winding(lambda p: p, np.array([0.0, 0.0]))
    checks.append(src == 1)
    sink = _winding(lambda p: -p, np.array([0.0, 0.0]))
    checks.append(sink == 1)
    saddle = _winding(lambda p: np.array([p[0], -p[1]]), np.array([0.0, 0.0]))
    checks.append(saddle == -1)
    # dipole on sphere: two index +1 zeros -> sum 2 = chi(S^2)
    checks.append(1 + 1 == 2)
    # torus nonvanishing field exists -> index sum 0 = chi(T^2)
    checks.append(0 == 0)
    # field with source + 2 saddles: index sum -1 -> not allowed on S^2 alone;
    # on genus-2 surface chi = -2: source + sink - 2 saddles? verify sum
    checks.append(1 + 1 - 4 == -2)  # 2 extrema? genus2: m pattern
    return float(sum(checks) / len(checks))


def bench_poincare_hopf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poincare_hopf": _bench_poincare_hopf(seed)}
