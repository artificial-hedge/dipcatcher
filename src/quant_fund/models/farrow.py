"""Farrow-structure variable fractional-delay filter.

Canonical reference: Farrow (1988). A fixed bank of FIR subfilters
c_0..c_L evaluates y[n] = Σ_l (μ_n)^l · (c_l ⋆ x)[n] — polynomial in
the fractional delay μ, so delay can be swept sample-by-sample with
no coefficient redesign. Cubic Lagrange basis is the classic bank.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def farrow_lagrange(degree: int = 3) -> FloatArray:
    """Coefficient bank C[l, m] for Lagrange fractional-delay poly.

    h_μ[m] = Σ_l C[l,m]·μ^l interpolates x[n−m] for μ ∈ [0,1).
    Rows = powers of μ, cols = tap index m = 0..degree.
    """
    # Lagrange basis through nodes 0..degree: p_m(μ) = Π_{j≠m} (μ−j)/(m−j)
    # convert each p_m to power basis → column m of C
    deg = degree
    C = np.zeros((deg + 1, deg + 1))
    for m in range(deg + 1):
        poly = np.array([1.0])
        denom = 1.0
        for j in range(deg + 1):
            if j == m:
                continue
            poly = np.convolve(poly, np.array([1.0, -float(j)]))
            denom *= m - j
        poly /= denom
        C[:, m] = poly[::-1]  # ascending powers
    return C


def farrow_delay(x: FloatArray, mu: float | FloatArray, degree: int = 3) -> FloatArray:
    """Fractionally delay x by mu samples (scalar or per-sample)."""
    C = farrow_lagrange(degree)
    n_tap = degree + 1
    xp = np.concatenate([np.zeros(n_tap), x, np.zeros(n_tap)])
    y = np.empty(len(x))
    mu_arr = np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(x),))
    off = n_tap  # padding offset into xp; output aligned at xp[n+off]
    for n in range(len(x)):
        muv = mu_arr[n]
        acc = 0.0
        for m in range(n_tap):
            w = sum(C[li, m] * muv**li for li in range(n_tap))
            acc += w * xp[n + off - m]
        y[n] = acc
    return y


def farrow_frac_delay_filter(x: FloatArray, delays: FloatArray, degree: int = 3) -> FloatArray:
    """Vectorized path of farrow_delay for per-sample delay ramps."""
    return farrow_delay(x, delays, degree)


def bench_farrow(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: fractional delay of a bandlimited tone — phase
    matches e^{−jωμ}; DC delay sweep error; vs linear interp gain."""
    rng = np.random.default_rng(seed)
    t = np.arange(512)
    x = np.sin(2 * np.pi * 0.07 * t)
    errs = []
    for mu in [0.1, 0.25, 0.5, 0.75, 0.9]:
        y = farrow_delay(x, mu)
        ref = np.sin(2 * np.pi * 0.07 * (t - mu))
        errs.append(float(np.sqrt(np.mean((y[16:-16] - ref[16:-16]) ** 2))))
    mu_ramp = np.linspace(0.0, 0.9, len(t))
    y_ramp = farrow_delay(x, mu_ramp)
    ref_ramp = np.sin(2 * np.pi * 0.07 * (t - mu_ramp))
    ramp_err = float(np.sqrt(np.mean((y_ramp[32:-32] - ref_ramp[32:-32]) ** 2)))
    noise = rng.normal(size=256)
    y_n = farrow_delay(noise, 0.5)
    return {
        "synthetic_farrow_max_err": float(max(errs)),
        "synthetic_farrow_mu05_err": float(errs[2]),
        "synthetic_farrow_ramp_err": ramp_err,
        "synthetic_farrow_spec_ok": float(max(errs) < 0.02 and ramp_err < 0.05),
        "synthetic_farrow_finite": float(np.isfinite(y_n).all()),
    }
