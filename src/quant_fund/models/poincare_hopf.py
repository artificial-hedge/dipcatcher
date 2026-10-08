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
    # dipole on sphere: source(+1) + sink(+1) index sum = 2 = chi(S^2)
    src_i = _winding(lambda p: p, np.array([0.0, 0.0]))
    checks.append(src_i + sink == 2)
    # torus nonvanishing field: constant field has winding 0 = chi(T^2)
    checks.append(_winding(lambda p: np.array([1.0, 0.5]), np.array([0.0, 0.0])) == 0)
    # genus-2: source + sink + 4 saddles = 1 + 1 - 4 = -2 = chi
    checks.append(src_i + sink + 4 * saddle == -2)
    if not all(checks):
        raise ValueError("Poincare-Hopf index-sum oracle failed")
    return float(sum(checks) / len(checks))


def bench_poincare_hopf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poincare_hopf": _bench_poincare_hopf(seed)}
