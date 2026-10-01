"""Hayashi-Yoshida lead-lag covariance on asynchronous ticks.

Regular-grid covariance breaks under non-synchronous trading
(the Epps effect: measured correlation collapses as sampling
frequency rises). The HY estimator sums products of
overlapping-observation increments — no grid — recovering the
true cross-asset covariance; shifting one series reveals
lead-lag direction.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure covariance recovery and
lead direction on generated asynchronous series — never market
evidence.

References:
- Hayashi, T., Yoshida, N. (2005). On covariance estimation of
  non-synchronously observed diffusion processes. *Bernoulli*
  11, 359-379.
- Epps, T. W. (1979). Comovements in stock prices in the very
  short run. *JASA* 74 — nonsynchronicity bias.
- Hoffmann, M., Rosenbaum, M., Yoshida, N. (2013). Estimation of
  the lead-lag parameter from non-synchronous data. *Bernoulli*
  19 — shifted-HY lead-lag contrast.
- de Jong, F., Nijman, T. (1997). High frequency analysis of
  lead-lag relationships. *J. Empirical Finance* 4.

Composition: pure numpy — interval-overlap increment products,
shift-scan for lead direction; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def hy_covariance(
    t1: FloatArray,
    p1: FloatArray,
    t2: FloatArray,
    p2: FloatArray,
) -> float:
    """HY covariance between two irregularly observed series.

    ``t1``/``p1`` and ``t2``/``p2`` are monotone timestamps and
    prices (log). Increments are formed on each series' own
    intervals; products count only when intervals overlap."""
    tt1 = np.asarray(t1, dtype=np.float64).ravel()
    tt2 = np.asarray(t2, dtype=np.float64).ravel()
    pp1 = np.asarray(p1, dtype=np.float64).ravel()
    pp2 = np.asarray(p2, dtype=np.float64).ravel()
    if tt1.size != pp1.size or tt2.size != pp2.size:
        raise ValueError("t/p length mismatch")
    if tt1.size < 30 or tt2.size < 30:
        raise ValueError("need >=30 ticks per series")
    if np.any(np.diff(tt1) <= 0) or np.any(np.diff(tt2) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if not np.all(np.isfinite(tt1)) or not np.all(np.isfinite(pp1)):
        raise ValueError("finite inputs required")
    if not np.all(np.isfinite(tt2)) or not np.all(np.isfinite(pp2)):
        raise ValueError("finite inputs required")

    d1 = np.diff(pp1)
    a1, b1 = tt1[:-1], tt1[1:]
    d2 = np.diff(pp2)
    a2, b2 = tt2[:-1], tt2[1:]
    # overlap of (a1,b1] × (a2,b2]: max(a) < min(b)
    cov = 0.0
    j = 0
    for i in range(d1.size):
        while j < d2.size and b2[j] <= a1[i]:
            j += 1
        k = j
        while k < d2.size and a2[k] < b1[i]:
            cov += d1[i] * d2[k]
            k += 1
    return float(cov)


def lead_lag_scan(
    t1: FloatArray,
    p1: FloatArray,
    t2: FloatArray,
    p2: FloatArray,
    max_shift: float = 5.0,
    n_shifts: int = 21,
) -> dict[str, float]:
    """Shift series 2 by δ and recompute HY covariance; the argmax
    |cov| shift exposes lead direction (Hoffmann-Rosenbaum-Yoshida).

    Positive δ* means series 2 must be delayed — series 1 leads."""
    tt1 = np.asarray(t1, dtype=np.float64).ravel()
    tt2 = np.asarray(t2, dtype=np.float64).ravel()
    if tt1.size < 30 or tt2.size < 30:
        raise ValueError("need >=30 ticks per series")
    if not (1.0 <= max_shift <= 50.0):
        raise ValueError("max_shift in 1..50")
    span = max(float(tt1[-1] - tt1[0]), float(tt2[-1] - tt2[0]))
    if max_shift >= 0.5 * span:
        raise ValueError("max_shift too large vs span")

    cov0 = hy_covariance(tt1, np.asarray(p1), tt2, np.asarray(p2))
    shifts = np.linspace(-max_shift, max_shift, n_shifts)
    covs = np.array([hy_covariance(tt1, np.asarray(p1), tt2 + s, np.asarray(p2)) for s in shifts])
    i_star = int(np.argmax(np.abs(covs)))
    delta_star = float(shifts[i_star])
    cov_star = float(covs[i_star])
    # contrast: |cov at best shift| vs |cov at 0|
    contrast = abs(cov_star) - abs(cov0)
    return {
        "cov0": cov0,
        "delta_star": delta_star,
        "cov_star": cov_star,
        "contrast": float(contrast),
        "leader": float(np.sign(delta_star)),
        "n_ticks1": float(tt1.size),
        "n_ticks2": float(tt2.size),
    }


def synth_async(
    t_span: float = 100.0,
    n1: int = 300,
    n2: int = 200,
    lag: float = 2.0,
    rho: float = 0.8,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Two Poisson-arrival price processes sharing a latent
    Brownian factor; series 2 lags series 1 by ``lag`` time units."""
    rng = np.random.default_rng(seed)
    t1 = np.sort(rng.uniform(0, t_span, n1))
    t2 = np.sort(rng.uniform(0, t_span, n2))
    # latent efficient price on a fine grid
    g = 2000
    gt = np.linspace(0, t_span * 1.2, g)
    dw = rng.normal(0.0, np.sqrt(t_span * 1.2 / g), g)
    latent = np.cumsum(dw)
    idio1 = np.cumsum(rng.normal(0.0, np.sqrt(t_span * 1.2 / g) * 0.4, g))
    idio2 = np.cumsum(rng.normal(0.0, np.sqrt(t_span * 1.2 / g) * 0.4, g))

    def sample(tt: FloatArray, offset: float) -> FloatArray:
        idx = np.clip(np.searchsorted(gt, tt + offset), 0, g - 1)
        return latent[idx]

    p1 = rho * sample(t1, 0.0) + idio1[np.clip(np.searchsorted(gt, t1), 0, g - 1)]
    p2 = rho * sample(t2, lag) + idio2[np.clip(np.searchsorted(gt, t2), 0, g - 1)]
    return {"t1": t1, "p1": p1, "t2": t2, "p2": p2, "lag": np.array([lag])}


def bench_lead_lag(seed: int = 20261231 + 235) -> dict[str, float]:
    """Lead-lag self-check: HY recovers positive covariance under
    asynchronicity and the shift scan finds the true lag
    direction. All ``synthetic_*``."""
    d = synth_async(lag=2.0, seed=seed)
    out = lead_lag_scan(
        np.asarray(d["t1"]),
        np.asarray(d["p1"]),
        np.asarray(d["t2"]),
        np.asarray(d["p2"]),
        max_shift=6.0,
    )
    d0 = synth_async(lag=0.0, seed=seed + 1)
    out0 = lead_lag_scan(
        np.asarray(d0["t1"]),
        np.asarray(d0["p1"]),
        np.asarray(d0["t2"]),
        np.asarray(d0["p2"]),
        max_shift=6.0,
    )
    out_b = lead_lag_scan(
        np.asarray(d["t1"]),
        np.asarray(d["p1"]),
        np.asarray(d["t2"]),
        np.asarray(d["p2"]),
        max_shift=6.0,
    )

    ds = float(out["delta_star"])
    return {
        "synthetic_cov": float(out["cov0"]),
        "synthetic_cov_star": float(out["cov_star"]),
        "synthetic_delta_star": ds,
        "synthetic_delta_err": float(abs(ds - 2.0)),
        "synthetic_contrast": float(out["contrast"]),
        "synthetic_delta_indep": float(abs(float(out0["delta_star"]))),
        "synthetic_detects": float(
            float(out["cov_star"]) > 0.0 and ds > 0.5 and abs(ds - 2.0) < 2.0
        ),
        "synthetic_determinism": float(
            ds == float(out_b["delta_star"]) and float(out["cov0"]) == float(out_b["cov0"])
        ),
    }
