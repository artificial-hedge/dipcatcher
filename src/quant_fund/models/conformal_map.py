"""Mobius maps are conformal: angles between curves preserved (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def mobius(z: np.ndarray, a: complex, b: complex, c: complex, d: complex) -> np.ndarray:
    return np.asarray((a * z + b) / (c * z + d), dtype=np.complex128)


def _angle_between(v1: complex, v2: complex) -> float:
    w = complex(v1) / abs(complex(v1)) * np.conj(complex(v2) / abs(complex(v2)))
    return float(np.arccos(np.clip(w.real, -1, 1)))


def _tangent(samples: np.ndarray, i: int) -> complex:
    return complex(samples[i + 1] - samples[i - 1])


def _bench_conformal_map(seed: int = 0) -> float:
    checks = []
    # curves: horizontal line and vertical line crossing at z0 = 0.5
    t = np.linspace(-0.4, 0.4, 2001)
    c1 = 0.5 + t + 0j * t  # horizontal through (0.5, 0)
    c2 = 0.5 + 1j * t  # vertical through (0.5, 0)
    i = len(t) // 2 - 1  # center index where t ~ 0
    ang_before = _angle_between(_tangent(c1, i), _tangent(c2, i))
    a, b, c, d = 1.0, 0.0, 1.0, 1.0  # w = z/(z+1), pole at -1 away from 0.5
    w1 = mobius(c1, a, b, c, d)
    w2 = mobius(c2, a, b, c, d)
    ang_after = _angle_between(_tangent(w1, i), _tangent(w2, i))
    checks.append(abs(ang_before - np.pi / 2) < 1e-9)
    checks.append(abs(ang_after - ang_before) < 1e-3)
    # w = (z - i)/(z + i) maps real axis to unit circle
    x = np.linspace(-100, 100, 2001)
    w = mobius(x + 0j, 1.0, -1j, 1.0, 1j)
    checks.append(np.allclose(np.abs(w), 1.0, atol=1e-9))
    # w = 1/z maps circle |z-1|=1 (through 0) to a line
    th = np.linspace(0.01, 2 * np.pi - 0.01, 4000)
    zc = 1.0 + np.exp(1j * th)
    wc = 1.0 / zc
    checks.append(float(np.std(np.real(wc))) < 1e-9)  # Re = 1/2 line
    # derivative of Mobius nonzero in its domain (conformal)
    z0 = 0.5 + 0j
    der = (a * d - b * c) / (c * z0 + d) ** 2
    checks.append(abs(der) > 1e-9)
    return float(sum(checks) / len(checks))


def bench_conformal_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conformal_map": _bench_conformal_map(seed)}
