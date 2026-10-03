"""Andreasen-Huge single-step arbitrage-free vol interpolation.

Given call prices at discrete strikes, solve the coupled system
C(K_i) consistent with a piecewise-constant local vol via one
implicit step per strike gap (the AH 'ZABR-style' step simplified to
local-vol). Bench: prices are arb-free (nonnegative vertical +
butterfly spreads) and recover the generating BS prices when the
local vol is flat.
"""

import numpy as np
from scipy.stats import norm


def _bs(s: float, k: float, t: float, sig: float) -> float:
    d1 = (np.log(s / k) + 0.5 * sig**2 * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return float(s * norm.cdf(d1) - k * norm.cdf(d2))


def _ah_step(s0: float, ks: np.ndarray, sigs: np.ndarray, t: float) -> np.ndarray:
    """Implicit solve per strike on a local-vol grid (Andreasen-Huge
    style): tridiagonal substeps from the intrinsic profile with vol
    given by the piecewise sigma at each K."""
    nk = len(ks)
    # grid spanning strikes
    grid = np.linspace(0.3 * s0, 3 * s0, 4 * nk + 1)
    dt = t
    out = np.zeros(nk)
    for j in range(nk):
        v0 = np.maximum(grid - ks[j], 0.0)
        sg = np.interp(grid, ks, sigs)
        dg = grid[1] - grid[0]
        from scipy.linalg import solve_banded

        nsub = 8
        sdt = dt / nsub
        al = sdt * 0.5 * sg[1:-1] ** 2 * grid[1:-1] ** 2 / dg**2
        di = -sdt * sg[1:-1] ** 2 * grid[1:-1] ** 2 / dg**2
        m = len(grid) - 2
        bands = np.zeros((3, m))
        bands[0, 1:] = -al[:-1]
        bands[1] = 1 - di
        bands[2, :-1] = -al[1:]
        v1 = v0.copy()
        for _ in range(nsub):
            rhs = v1[1:-1].copy()
            rhs[-1] += al[-1] * (grid[-1] - ks[j])
            v1[1:-1] = solve_banded((1, 1), bands, rhs)
            v1[-1] = grid[-1] - ks[j]
        out[j] = np.interp(s0, grid, v1)
    return out


def bench_andreasen_huge(seed: int = 5809) -> dict[str, float]:
    _ = seed
    s0, t = 100.0, 0.5
    ks = np.linspace(80, 120, 9)
    sig_flat = np.full(len(ks), 0.2)
    px_ah = _ah_step(s0, ks, sig_flat, t)
    px_bs = np.array([_bs(s0, k, t, 0.2) for k in ks])
    # arb checks: decreasing in K, convex
    mono = float(np.all(np.diff(px_ah) <= 1e-8))
    conv = float(np.all(np.diff(px_ah, 2) >= -1e-6))
    err = float(np.max(np.abs(px_ah - px_bs)))
    return {
        "synthetic_ah_mono": mono,
        "synthetic_ah_convex": conv,
        "synthetic_ah_max_err_vs_bs": err,
    }
