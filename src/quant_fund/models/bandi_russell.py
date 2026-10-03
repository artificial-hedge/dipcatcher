"""Bandi-Russell separation of microstructure noise from variance.

References
----------
- Bandi, F.M. & Russell, J.R. (2008). "Separating Microstructure
  Noise from Volatility." *Journal of Financial Economics* 79(3),
  655-692.
- Hansen, P.R. & Lunde, A. (2006). "Realized Variance and Market
  Microstructure Noise." *JBES* 24(2), 127-161.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
The observed log price is efficient price plus iid level noise
``eta z_t``, so observed returns carry MA(1) noise of variance
``2 eta^2`` each. The finest-grid realized variance is biased by
``2 T eta^2``; sparse-grid RV over ``T/m`` blocks is biased by
``2 eta^2 T/m`` (each block return holds only two noise draws),
while its discretization variance is ``2 IV^2 m/T`` — the
Bandi-Russell trade-off that pins the optimal frequency
``m* = (4 eta^4 T^3 / IV^2)^(1/3)``. We estimate
``eta^2 = RV_full/(2T)`` and a bias-corrected pilot ``IV``, solve
``m*`` on the integer grid clipped to ``[2, T/8]``, and report
``rv_br = RV(m*) - 2 eta2 T/m*`` plus the naive ``RV_full`` for
the improvement comparison the synth must exhibit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def noise_variance(r: FloatArray) -> float:
    """Bandi-Russell noise-variance estimate ``RV_full / (2T)``."""
    rr = np.asarray(r, dtype=np.float64)
    if rr.ndim != 1 or rr.size < 20 or not np.all(np.isfinite(rr)):
        raise ValueError("bad returns")
    return float(np.sum(rr * rr) / (2.0 * rr.size))


def _sparse_rv(r: FloatArray, m: int) -> float:
    """RV on the m-sparse grid: block returns are m-step sums."""
    n_full = r.size // m
    if n_full < 2:
        return float("nan")
    blocks = r[: n_full * m].reshape(n_full, m).sum(axis=1)
    return float(np.sum(blocks * blocks))


def optimal_m(t: int, eta2: float, iv: float) -> int:
    """BR optimal frequency ``(4 eta^4 T^3 / IV^2)^(1/3)`` clipped."""
    if iv <= 0.0 or eta2 <= 0.0 or t < 16:
        raise ValueError("bad inputs")
    m_star = (4.0 * eta2 * eta2 * t**3 / (iv * iv)) ** (1.0 / 3.0)
    m_grid = np.clip(np.arange(max(2, int(m_star * 0.4)), int(m_star * 2.5) + 2), 2, t // 8)
    # exact integer argmin of the MSE expansion (bias^2 + variance)
    mse = (2.0 * eta2 * t / m_grid) ** 2 + 2.0 * iv * iv * m_grid / t
    return int(m_grid[int(np.argmin(mse))])


def br_realized_variance(r: FloatArray) -> dict[str, float]:
    """Bandi-Russell noise-robust realized variance.

    ``eta2``: iid level-noise variance; ``m_star``: optimal sparse
    frequency; ``rv_star``: sparse RV; ``rv_br``: bias-corrected
    ``rv_star - 2 eta2 T/m*``; ``rv_full``: naive fine-grid RV.
    """
    rr = np.asarray(r, dtype=np.float64)
    if rr.ndim != 1 or rr.size < 60 or not np.all(np.isfinite(rr)):
        raise ValueError("bad returns")
    t = rr.size
    rv_full = float(np.sum(rr * rr))
    eta2 = rv_full / (2.0 * t)
    # pilot IV: moderately sparse grid minus its own noise bias
    m_pilot = max(2, t // 100)
    iv_pilot = max(_sparse_rv(rr, m_pilot) - 2.0 * eta2 * t / m_pilot, rv_full * 0.005)
    m = optimal_m(t, eta2, iv_pilot)
    rv_star = _sparse_rv(rr, m)
    bias = 2.0 * eta2 * t / m
    rv_br = rv_star - bias
    return {
        "eta2": eta2,
        "iv_pilot": iv_pilot,
        "m_star": float(m),
        "rv_star": rv_star,
        "rv_br": rv_br,
        "rv_full": rv_full,
        "bias_sub": bias,
        "t": float(t),
    }


def synth_br(
    seed: int = 20261231 + 302,
    t: int = 4000,
    iv: float = 1.0,
    eta: float = 0.02,
) -> dict[str, float | FloatArray]:
    """SYNTHETIC efficient prices + iid level noise -> MA(1) noise."""
    rng = np.random.default_rng(seed)
    sig = np.sqrt(iv / t)
    eff = np.cumsum(sig * rng.standard_normal(t + 1))
    obs = eff + eta * rng.standard_normal(t + 1)
    r = np.diff(obs)
    return {
        "r": r,
        "iv_true": iv,
        "eta2_true": eta * eta,
    }


def bench_bandi_russell(seed: int = 20261231 + 302) -> dict[str, float]:
    """Wave-52 self-check: BR variance beats naive RV materially."""
    d = synth_br(seed=seed)
    r = br_realized_variance(np.asarray(d["r"]))
    iv = float(d["iv_true"])
    err_br = abs(float(r["rv_br"]) - iv) / iv
    err_naive = abs(float(r["rv_full"]) - iv) / iv
    ratio = err_naive / max(err_br, 1e-9)
    ok = err_br < 0.35 and ratio > 2.0 and 1.0 < r["m_star"] < r["t"] / 4.0
    return {
        "rv_br": float(r["rv_br"]),
        "iv_true": iv,
        "err_br": err_br,
        "err_naive": err_naive,
        "improvement": ratio,
        "m_star": float(r["m_star"]),
        "eta2_hat": float(r["eta2"]),
        "eta2_true": float(d["eta2_true"]),
        "score": float(ok),
    }
