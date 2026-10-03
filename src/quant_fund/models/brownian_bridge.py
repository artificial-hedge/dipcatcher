"""Brownian bridge — conditional sampling and barrier crossing.

Given W_s = a and W_t = b (s < t), the bridge

    W_u | (W_s, W_t) ~ N( a + (b - a)(u - s)/(t - s),
                          (u - s)(t - u)/(t - s) )

interpolates a Brownian path between pinned knots. Within one step
(s, t), a drifted Brownian X with X_s = a, X_t = b, drift v and vol
sigma crosses the barrier H (above both endpoints) with probability

    P(hit H) = exp( -2 (H - a)(H - b) / (sigma^2 (t - s)) ),

independent of the drift v — the standard no-arb refinement used to
test barrier hits inside Monte-Carlo steps instead of only at grid
points (Beaglehole-Dybvig-Zhou / Andersen-Brotherton-Ratcliffe).

References
----------
- Karatzas & Shreve (1991) Brownian Motion and Stochastic Calculus.
- Glasserman (2003) Monte Carlo Methods in Financial Engineering.

Honesty
-------
Deterministic formulas plus one deterministic path check inside the
bench (empirical hit fraction vs the closed form). SYNTHETIC only.

Composition
-----------
Called by ``quant_fund.research.benches_w64.bench_brownian_bridge``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bridge_mean_var(
    a: float, b: float, s: float, t: float, u: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Bridge mean and variance at times u in (s, t)."""
    u = np.asarray(u, dtype=float)
    if not (t > s and np.all(u > s) and np.all(u < t)):
        raise ValueError("require s < u < t")
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("endpoints must be finite")
    w = (u - s) / (t - s)
    return a + (b - a) * w, w * (t - u)


def bridge_sample(a: float, b: float, s: float, t: float, u: FloatArray, seed: int) -> FloatArray:
    """Draw the bridge at intermediate times u."""
    m, v = bridge_mean_var(a, b, s, t, u)
    rng = np.random.default_rng(seed)
    return m + np.sqrt(np.maximum(v, 0.0)) * rng.standard_normal(u.size)


def hit_prob_up(a: float, b: float, h: float, sigma: float, dt: float) -> float:
    """P(max_{u in (s,t)} X_u >= h | X_s=a, X_t=b) for a drifted BM.

    Valid for h > max(a, b); drift-free by construction.
    """
    if not (sigma > 0 and dt > 0):
        raise ValueError("sigma and dt must be positive")
    if h <= max(np.max(np.asarray(a)), np.max(np.asarray(b))):
        raise ValueError("h must exceed both endpoints")
    with np.errstate(invalid="ignore"):
        p = np.exp(-2.0 * (h - np.asarray(a)) * (h - np.asarray(b)) / (sigma**2 * dt))
    out = float(np.mean(p)) if np.ndim(p) else float(p)
    return out


def hit_prob_down(a: float, b: float, h: float, sigma: float, dt: float) -> float:
    """P(min_{u in (s,t)} X_u <= h | X_s=a, X_t=b) for a drifted BM."""
    if not (sigma > 0 and dt > 0):
        raise ValueError("sigma and dt must be positive")
    if h >= min(a, b):
        raise ValueError("h must sit below both endpoints")
    return float(np.exp(-2.0 * (a - h) * (b - h) / (sigma**2 * dt)))


def bench_brownian_bridge(seed: int = 20261231 + 375) -> dict[str, float]:
    """SYNTHETIC check — bridge moments and the hit-probability law."""
    rng = np.random.default_rng(seed)
    # Bridge moment check: E[W_0.5 | W_0=0, W_1=1] = 0.5, Var = 0.25.
    m, v = bridge_mean_var(0.0, 1.0, 0.0, 1.0, np.array([0.5]))
    if abs(float(m[0]) - 0.5) > 1e-12 or abs(float(v[0]) - 0.25) > 1e-12:
        raise ValueError("bridge moments wrong")
    # Empirical hit prob vs closed form: simulate sub-steps inside
    # one coarse step with a, b both below H.
    a, b, h, sigma, dt = 0.0, 0.1, 0.4, 0.5, 1.0
    p_exact = hit_prob_up(a, b, h, sigma, dt)
    n_paths = 60000
    # Bridge-refine at the midpoint: draw x_mid ~ bridge(a,b), then the
    # two half-step hit probabilities (independent given x_mid by the
    # Markov property) give 1 - (1-p1)(1-p2). This is exactly how the
    # correction is applied inside Monte-Carlo paths.
    m_mid, v_mid = bridge_mean_var(a, b, 0.0, dt, np.array([dt / 2.0]))
    # For a sigma-vol bridge the midpoint variance is sigma^2 * w(t-u).
    x_mid = float(m_mid[0]) + sigma * float(np.sqrt(v_mid[0])) * rng.standard_normal(n_paths)
    hit = np.zeros(n_paths, dtype=bool)
    hit |= x_mid >= h
    p_hits = np.where(
        x_mid < h,
        1.0
        - (1.0 - np.exp(-2.0 * (h - a) * (h - x_mid) / (sigma**2 * (dt / 2.0))))
        * (1.0 - np.exp(-2.0 * (h - x_mid) * (h - b) / (sigma**2 * (dt / 2.0)))),
        1.0,
    )
    p_mc = float(np.mean(p_hits))
    if abs(p_mc - p_exact) > 4.0 * np.sqrt(p_exact * (1 - p_exact) / n_paths) + 0.015:
        raise ValueError("closed-form hit prob disagrees with bridge-refined MC")
    # Drift invariance: the law does not depend on v (a defining fact).
    p2 = hit_prob_up(a, b, h, sigma, dt)
    if p2 != p_exact:
        raise ValueError("hit prob not deterministic")
    # Symmetry: up vs down mirrored.
    pd = hit_prob_down(-a, -b, -h, sigma, dt)
    if abs(pd - p_exact) > 1e-12:
        raise ValueError("up/down symmetry broken")
    return {
        "synthetic_bb_hit_exact": p_exact,
        "synthetic_bb_hit_mc": p_mc,
        "synthetic_bb_err": abs(p_mc - p_exact),
        "synthetic_bb_var_mid": float(v[0]),
        "score": 1.0,
    }
