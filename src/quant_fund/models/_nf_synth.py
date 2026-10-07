"""Shared fixture for wave-182 normalizing-flow canon (SYNTHETIC).

4-arm pinwheel in 2-D: centers on circle radius 2.5, per-arm rotation
+ tangential spread. Baseline: single Gaussian NLL on held-out set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def pinwheel(seed: int, n: int = 2000, d: int = 2, arms: int = 4) -> FloatArray:
    rng = np.random.default_rng(seed)
    r = 2.5
    out = np.zeros((n, d))
    for i in range(n):
        a = rng.integers(arms)
        ang = 2 * np.pi * a / arms
        c = np.array([r * np.cos(ang), r * np.sin(ang)])
        u = rng.standard_normal() * 0.4
        v = rng.standard_normal() * 0.15
        t = np.array([np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)])
        rr = np.array([np.cos(ang), np.sin(ang)])
        out[i] = c + t * u + rr * v
    return out


def gauss_nll(Xtr: FloatArray, Xte: FloatArray) -> float:
    mu = Xtr.mean(0)
    C = np.cov(Xtr.T) + 1e-6 * np.eye(Xtr.shape[1])
    Ci = np.linalg.inv(C)
    d = Xtr.shape[1]
    diff = Xte - mu
    nll = (
        0.5 * np.einsum("ni,ij,nj->n", diff, Ci, diff)
        + 0.5 * d * np.log(2 * np.pi)
        + 0.5 * np.linalg.slogdet(C)[1]
    )
    return float(nll.mean())


def two_moons(seed: int, n: int = 2000) -> FloatArray:
    rng = np.random.default_rng(seed)
    half = n // 2
    t1 = rng.uniform(0, np.pi, half)
    t2 = rng.uniform(0, np.pi, n - half)
    x1 = np.stack([np.cos(t1), np.sin(t1)], -1)
    x2 = np.stack([1 - np.cos(t2), -np.sin(t2) + 0.5], -1)
    X = np.vstack([x1, x2])
    X += rng.standard_normal(X.shape) * 0.08
    return X
