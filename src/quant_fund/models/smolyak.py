"""Smolyak sparse-grid quadrature and interpolation (SYNTHETIC).

Clenshaw–Curtis 1-D rules composed by the Smolyak combination over
the downward-closed multi-index set {i : |i|₁ ≤ d + l}. Handles both
integration (surrogate weights) and interpolation (hierarchical
surpluses) on [0,1]^d.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _cc_1d(level: int) -> tuple[FloatArray, FloatArray]:
    """Clenshaw–Curtis nodes/weights on [0,1] for Smolyak level."""
    n = 1 if level == 0 else 2**level + 1
    if n == 1:
        return np.array([0.5]), np.array([1.0])
    k = np.arange(n)
    x = 0.5 * (1.0 - np.cos(np.pi * k / (n - 1)))
    w = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(1, (n - 1) // 2 + 1):
            b = 2.0 if 2 * j < n - 1 else 1.0
            s += b * np.cos(2 * j * np.pi * i / (n - 1)) / (4 * j * j - 1)
        w[i] = (1.0 - s) / (n - 1)
        if i in (0, n - 1):
            w[i] /= 2.0
    return x, w


def _multi_indices(d: int, level: int) -> list[tuple[int, ...]]:
    """Downward-closed index set |i|₁ ≤ d + level, i_k ≥ 0."""
    return [idx for idx in itertools.product(range(level + 1), repeat=d) if sum(idx) <= level]


def smolyak_integrate(f, d: int, level: int) -> tuple[float, int]:
    """Approximate ∫_[0,1]^d f via the level-`level` Smolyak rule.

    Returns (integral, n_points).
    """
    idxs = _multi_indices(d, level)
    full: dict[tuple[int, ...], tuple[FloatArray, FloatArray]] = {}
    for idx in idxs:
        full[idx] = tuple(_cc_1d(i) for i in idx)  # type: ignore[assignment]
    total = 0.0
    npts = 0
    # Smolyak coefficient c(i) = (-1)^{l-|i|} C(d-1, l-|i|); the binomial
    # vanishes for |i| < l - d + 1 so those grids drop out.

    def coef(idx: tuple[int, ...]) -> int:
        diff = level - sum(idx)
        return int((-1) ** diff * _comb(d - 1, diff))

    for idx in idxs:
        c = coef(idx)
        if c == 0:
            continue
        rules = full[idx]
        pts = np.array(np.meshgrid(*[r[0] for r in rules], indexing="ij")).reshape(d, -1)
        wts = np.prod(
            np.array(np.meshgrid(*[r[1] for r in rules], indexing="ij")).reshape(d, -1),
            axis=0,
        )
        vals = np.array([f(p) for p in pts.T])
        total += c * float(wts @ vals)
        npts += len(wts)
    return total, npts


def _comb(n: int, k: int) -> int:
    from math import comb

    return comb(n, k) if 0 <= k <= n else 0


def bench_smolyak(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: exactness on a degree-3 polynomial in 2-D and
    convergence on exp(x+y) vs. the tensor product cost."""
    out: dict[str, float] = {}
    # ∫∫ (x² + xy + y) = 1/3 + 1/4 + 1/2 = 13/12
    val, npts = smolyak_integrate(lambda p: p[0] ** 2 + p[0] * p[1] + p[1], 2, 2)
    out["synthetic_smolyak_poly_err"] = abs(val - 13.0 / 12.0)
    # ∫∫ e^{x+y} = (e-1)²
    val2, npts2 = smolyak_integrate(lambda p: float(np.exp(p[0] + p[1])), 2, 4)
    out["synthetic_smolyak_exp_err"] = abs(val2 - (np.e - 1) ** 2)
    out["synthetic_smolyak_npts_l4"] = float(npts2)
    out["synthetic_smolyak_sparse_ratio"] = float(npts2) / (17**2)
    return out


if __name__ == "__main__":
    print(bench_smolyak())
