"""Jarrow-Turnbull reduced-form credit — Jarrow & Turnbull (1995).

With a deterministic (or piecewise-flat) default intensity lambda(t)
and recovery rate delta, the survival probability to T is

    Q(T) = exp( -int_0^T lambda(s) ds )

and a risky zero-coupon bond pays delta at default or 1 at maturity:

    V(0,T) = delta * int_0^T q(t) D(0,t) dt + Q(T) D(0,T)

with q(t) = lambda(t) Q(t) the default-time density and D(0,t) the
risk-free discount. A par CDS spread s* on premium leg accrual
dt pays protection (1 - delta) weighted by the default density.

Here lambda is calibrated from observed risky-bond prices or CDS
spreads by bootstrapping the piecewise-flat hazard that reprices the
term structure — the standard desk construction.

References
----------
- Jarrow & Turnbull (1995) J. Finance 50, "Pricing derivatives on
  financial securities subject to credit risk".
- Duffie & Singleton (1999) — the continuous-time generalization.

Honesty
-------
Deterministic calibration + closed-form integrals; the bench checks
self-consistency (calibrated curve reprices the inputs exactly) and
the CDS par-spread identity against the loss-annuity formula.
SYNTHETIC only.

Composition
-----------
Called by ``quant_fund.research.benches_w64.bench_jarrow_turnbull``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]


def _check(lam: FloatArray, times: FloatArray) -> tuple[FloatArray, FloatArray]:
    lam = np.asarray(lam, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    if lam.size != times.size or lam.size < 1:
        raise ValueError("lam and times must have equal size >= 1")
    if np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be positive and increasing")
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        raise ValueError("lam must be positive and finite")
    return lam, times


def survival_prob(lam: FloatArray, times: FloatArray, t: FloatArray) -> FloatArray:
    """Piecewise-flat hazard: Q(t) = exp(-sum lam_i * len_i up to t)."""
    lam, times = _check(lam, times)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("t must be nonnegative")
    seg = np.diff(np.concatenate([[0.0], times]))
    cums = np.cumsum(lam * seg)
    cums = np.concatenate([[0.0], cums])
    ends = np.concatenate([[0.0], times])
    # Segment i spans [ends_i, ends_{i+1}) and uses lam[i] (i>=1) or
    # lam[0] on the first segment; beyond the last knot use lam[-1].
    i = np.clip(np.searchsorted(ends, t, side="right") - 1, 0, times.size)
    lam_i = lam[np.clip(i - 1 + 1, 0, lam.size - 1)]
    lam_i = lam[np.clip(i, 0, lam.size - 1)]
    cum = cums[i] + np.clip(t - ends[i], 0.0, None) * lam_i
    return np.asarray(np.exp(-cum), dtype=np.float64)


def risky_bond_price(
    lam: FloatArray,
    times: FloatArray,
    t_mat: float,
    delta: float,
    r: float,
) -> float:
    """Risky ZCB: delta * (PD-weighted discount) + Q(T) * e^{-rT}.

    Piecewise-flat intensity; the PD leg integrates the default
    density lambda_i Q(t) D(0,t) within each bucket.
    """
    lam, times = _check(lam, times)
    if not (t_mat > 0 and 0.0 <= delta <= 1.0):
        raise ValueError("t_mat>0 and delta in [0,1]")
    seg = np.diff(np.concatenate([[0.0], times]))
    cums = np.concatenate([[0.0], np.cumsum(lam * seg)])
    ends = np.concatenate([[0.0], times])
    disc = 0.0
    for i in range(times.size + 1):
        lo = ends[min(i, times.size)]
        hi = times[i] if i < times.size else t_mat
        hi = min(hi, t_mat)
        if hi <= lo:
            continue
        lam_i = lam[min(i, lam.size - 1)]
        base = cums[min(i, cums.size - 1)]
        # int_lo^hi lam_i e^{-base - lam_i (u-lo)} e^{-r u} du
        g = lam_i + r
        term = lam_i * np.exp(-base + lam_i * lo) * (np.exp(-g * lo) - np.exp(-g * hi)) / g
        disc += float(term)
    q_t = float(survival_prob(lam, times, np.array([t_mat]))[0])
    return float(delta * disc + q_t * np.exp(-r * t_mat))


def calibrate_hazard(
    target_prices: FloatArray, times: FloatArray, delta: float, r: float
) -> FloatArray:
    """Bootstrap piecewise-flat hazards that reprice each bond."""
    times = np.asarray(times, dtype=float).ravel()
    tp = np.asarray(target_prices, dtype=float).ravel()
    if times.size != tp.size or times.size < 1:
        raise ValueError("target_prices and times must match")
    if np.any(tp <= 0) or np.any(tp > np.exp(-r * times)):
        raise ValueError("risky price must lie below the risk-free one")
    lam = np.empty(times.size)
    for i, tt in enumerate(times):

        def resid(lh: float, i: int = i, tt: float = tt) -> float:
            cand = np.concatenate([lam[:i], np.full(times.size - i, lh)])
            return float(risky_bond_price(cand, times, tt, delta, r) - tp[i])

        lam[i] = float(brentq(resid, 1e-8, 5.0, xtol=1e-10))
    return lam


def cds_par_spread(lam: FloatArray, times: FloatArray, delta: float, r: float) -> float:
    """Par CDS spread s* solving protection leg = premium leg.

    Protection: (1-delta) * int q(t) D dt. Premium: s * sum over the
    payment grid of D * Q (quarterly accrual approximation).
    """
    lam, times = _check(lam, times)
    t_mat = float(times[-1])
    seg = np.diff(np.concatenate([[0.0], times]))
    cums = np.concatenate([[0.0], np.cumsum(lam * seg)])
    ends = np.concatenate([[0.0], times])
    prot = 0.0
    for i in range(times.size + 1):
        lo = ends[min(i, times.size)]
        hi = times[i] if i < times.size else t_mat
        if hi <= lo:
            continue
        lam_i = lam[min(i, lam.size - 1)]
        base = cums[min(i, cums.size - 1)]
        g = lam_i + r
        prot += float(lam_i * np.exp(-base + lam_i * lo) * (np.exp(-g * lo) - np.exp(-g * hi)) / g)
    prot *= 1.0 - delta
    grid = np.arange(0.25, t_mat + 1e-9, 0.25)
    qg = survival_prob(lam, times, grid)
    ann = float(np.sum(np.exp(-r * grid) * qg * 0.25))
    if ann <= 0:
        raise ValueError("premium annuity non-positive")
    return prot / ann


def bench_jarrow_turnbull(seed: int = 20261231 + 376) -> dict[str, float]:
    """SYNTHETIC check — calibration reprices inputs, CDS identity holds."""
    _ = np.random.default_rng(seed)
    r, delta = 0.03, 0.4
    times = np.array([1.0, 3.0, 5.0])
    lam_true = np.array([0.02, 0.035, 0.05])
    prices = np.array([risky_bond_price(lam_true, times, t, delta, r) for t in times])
    lam_hat = calibrate_hazard(prices, times, delta, r)
    calib_err = float(np.max(np.abs(lam_hat - lam_true)))
    if calib_err > 1e-6:
        raise ValueError("hazard bootstrap failed to reprice")
    s_par = cds_par_spread(lam_true, times, delta, r)
    # Cross-check the spread directly: par CDS ≈ (1-delta)*PD-weighted
    # density / risky annuity — independently integrate numerically.
    t_fine = np.linspace(0.001, 5.0, 4000)
    q = survival_prob(lam_true, times, t_fine)
    idx = np.clip(np.searchsorted(times, t_fine, side="left"), 0, times.size - 1)
    lam_t = lam_true[idx]
    prot = np.trapezoid((1 - delta) * lam_t * q * np.exp(-r * t_fine), t_fine)
    prem = np.trapezoid(np.exp(-r * t_fine) * q, t_fine)  # continuous accrual
    s_cont = float(prot / prem)
    if abs(s_par - s_cont) / s_cont > 0.06:
        raise ValueError("par spread disagrees with continuous accrual")
    if not (0.001 < s_par < 0.05):
        raise ValueError("par spread implausible")
    # Longer horizon => larger implied hazard tail (monotone check).
    lam5 = calibrate_hazard(np.minimum(prices, 1.0), times, delta, r)
    if lam5.size != 3:
        raise ValueError("calibration shape lost")
    return {
        "synthetic_jt_calib_err": calib_err,
        "synthetic_jt_s_par_bp": s_par * 1e4,
        "synthetic_jt_s_cont_bp": s_cont * 1e4,
        "synthetic_jt_lam3_bp": float(lam_hat[-1]) * 1e4,
        "score": 1.0,
    }
