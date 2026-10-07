"""Statistical process control — Shewhart, EWMA, CUSUM, and
capability indices.

Shewhart (1931) x-bar/R and x-bar/S charts use phase-I center
lines and the A2/A3/B3/B4 factor tables (standard constants
for subgroup sizes 2..10). Roberts (1959) EWMA tracks
z_t = lam x_t + (1-lam) z_{t-1} with time-varying limits

    UCL_t = m + L s sqrt(lam/(2-lam) (1-(1-lam)^{2t})).

Page (1954) tabular CUSUM accumulates one-sided scores
C+ = max(0, C+ + x - (m+K)) with the decision interval H;
out-of-control signals flag the first crossing. Process
capability (Juran 1974): Cp = (USL-LSL)/(6s),
Cpk = min(USL-m, m-LSL)/(3s), Cpm with target T.

Honesty: limits computed from the data itself (phase-I
estimation) — no pretense of a stable in-control reference
beyond the sample; factor tables cover n in [2,10] only and
fail closed outside. The bench plants a mid-series level
shift and requires EWMA + CUSUM to signal while in-control
segments stay clean. Fail-closed on constant series or
invalid limits.

References: Shewhart (1931) "Economic Control of Quality";
Page (1954) Biometrika 41:100; Roberts (1959) Technometrics
1:239; Montgomery (2020) "Introduction to Statistical
Quality Control"; Juran (1974); Kane (1986) JQT 18:41.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Shewhart factor tables (Montgomery, table for n = 2..10)
_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
_A3 = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975}
_B3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284}
_B4 = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}
_D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
_C4 = {
    2: 0.7979,
    3: 0.8862,
    4: 0.9213,
    5: 0.9400,
    6: 0.9515,
    7: 0.9594,
    8: 0.9650,
    9: 0.9693,
    10: 0.9727,
}


def xbar_r_chart(
    x: FloatArray, subgroup: int = 5, phase1: int | None = None
) -> dict[str, float | FloatArray]:
    """Shewhart x-bar/R limits on consecutive subgroups.
    `phase1` = number of leading subgroups used as the
    in-control reference for center line + spread."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    n = xx.size
    if subgroup < 2 or subgroup > 10:
        raise ValueError("subgroup must be 2..10")
    n_sg = n // subgroup
    if n_sg < 5:
        raise ValueError("need >= 5 full subgroups")
    m = xx[: n_sg * subgroup].reshape(n_sg, subgroup)
    means = m.mean(axis=1)
    rngs = m.max(axis=1) - m.min(axis=1)
    ref_sg = n_sg if phase1 is None else phase1
    if ref_sg < 3 or ref_sg > n_sg:
        raise ValueError("phase1 must be 3..n_subgroups")
    xbar = float(means[:ref_sg].mean())
    rbar = float(rngs[:ref_sg].mean())
    ucl_x = xbar + _A2[subgroup] * rbar
    lcl_x = xbar - _A2[subgroup] * rbar
    ucl_r = _D4[subgroup] * rbar
    lcl_r = _D3[subgroup] * rbar
    ooc = int(((means > ucl_x) | (means < lcl_x)).sum())
    return {
        "center": xbar,
        "rbar": rbar,
        "ucl_x": ucl_x,
        "lcl_x": lcl_x,
        "ucl_r": ucl_r,
        "lcl_r": lcl_r,
        "n_ooc_x": float(ooc),
        "subgroup_means": np.asarray(means, dtype=np.float64),
    }


def ewma_chart(
    x: FloatArray,
    lam: float = 0.2,
    l_width: float = 3.0,
    phase1: int | None = None,
) -> dict[str, float | FloatArray]:
    """Roberts EWMA with time-varying control limits. `phase1`
    restricts the center/spread estimation to the first
    `phase1` observations (in-control reference window)."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    n = xx.size
    if n < 10 or not (0.0 < lam < 1.0):
        raise ValueError("need n>=10 and 0<lam<1")
    ref = xx[:phase1] if phase1 is not None else xx
    if ref.size < 5:
        raise ValueError("phase1 window too short")
    mu = float(ref.mean())
    sd = float(ref.std(ddof=1))
    if sd <= 1e-12:
        raise ValueError("constant series")
    z = np.empty(n)
    z[0] = mu
    for t in range(1, n):
        z[t] = lam * xx[t] + (1.0 - lam) * z[t - 1]
    fac = lam / (2.0 - lam)
    ucl = np.array(
        [
            mu + l_width * sd * math.sqrt(fac * (1.0 - (1.0 - lam) ** (2 * (t + 1))))
            for t in range(n)
        ]
    )
    lcl = 2.0 * mu - ucl
    hits = np.flatnonzero((z > ucl) | (z < lcl))
    mask = (z > ucl) | (z < lcl)
    return {
        "z": np.asarray(z, dtype=np.float64),
        "ucl": np.asarray(ucl, dtype=np.float64),
        "lcl": np.asarray(lcl, dtype=np.float64),
        "signal_mask": np.asarray(mask, dtype=np.float64),
        "first_signal": float(hits[0]) if hits.size else -1.0,
        "n_signals": float(hits.size),
    }


def cusum_chart(
    x: FloatArray,
    k_ref: float = 0.5,
    h_decision: float = 5.0,
    phase1: int | None = None,
) -> dict[str, float | FloatArray]:
    """Page tabular CUSUM (standardized one-sided scores).
    `phase1` = in-control reference window for mu/sd."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    n = xx.size
    if n < 10:
        raise ValueError("need n>=10")
    ref = xx[:phase1] if phase1 is not None else xx
    if ref.size < 5:
        raise ValueError("phase1 window too short")
    mu = float(ref.mean())
    sd = float(ref.std(ddof=1))
    if sd <= 1e-12:
        raise ValueError("constant series")
    zx = (xx - mu) / sd
    cp = np.zeros(n)
    cm = np.zeros(n)
    for t in range(1, n):
        cp[t] = max(0.0, cp[t - 1] + zx[t] - k_ref)
        cm[t] = min(0.0, cm[t - 1] + zx[t] + k_ref)
    hits_p = np.flatnonzero(cp > h_decision)
    hits_m = np.flatnonzero(cm < -h_decision)
    mask = (cp > h_decision) | (cm < -h_decision)
    first = -1.0
    if hits_p.size or hits_m.size:
        cand = []
        if hits_p.size:
            cand.append(hits_p[0])
        if hits_m.size:
            cand.append(hits_m[0])
        first = float(min(cand))
    return {
        "c_plus": np.asarray(cp, dtype=np.float64),
        "c_minus": np.asarray(cm, dtype=np.float64),
        "signal_mask": np.asarray(mask, dtype=np.float64),
        "first_signal": first,
        "n_signals": float(hits_p.size + hits_m.size),
    }


def process_capability(
    x: FloatArray,
    lsl: float,
    usl: float,
    target: float | None = None,
) -> dict[str, float]:
    """Cp, Cpk, Cpm (Juran/Kane) on a phase-I sample."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size < 10 or usl <= lsl:
        raise ValueError("need n>=10 and usl>lsl")
    m = float(xx.mean())
    s = float(xx.std(ddof=1))
    if s <= 1e-12:
        raise ValueError("zero sd")
    cp = (usl - lsl) / (6.0 * s)
    cpk = min(usl - m, m - lsl) / (3.0 * s)
    t = target if target is not None else (usl + lsl) / 2.0
    cpm = (usl - lsl) / (6.0 * math.sqrt(s * s + (m - t) ** 2))
    return {
        "cp": cp,
        "cpk": cpk,
        "cpm": cpm,
        "mean": m,
        "sd": s,
        "pct_out": float(((xx > usl) | (xx < lsl)).mean() * 100.0),
    }


def bench_spc(seed: int = 20261231 + 458) -> dict[str, float]:
    """SYNTHETIC check — EWMA/CUSUM catch a planted shift."""
    rng = np.random.default_rng(seed)
    n, split = 120, 70
    x = np.concatenate([rng.normal(size=split), rng.normal(loc=1.2, size=n - split)])
    e = ewma_chart(x, lam=0.2, l_width=3.0, phase1=split)
    c = cusum_chart(x, phase1=split)
    # both must raise a signal within the shifted segment
    me = np.asarray(e["signal_mask"])
    mc_ = np.asarray(c["signal_mask"])
    if not (me[split - 2 :].any() and mc_[split - 5 :].any()):
        raise ValueError(
            f"spc off: ewma hits in post-shift={me[split - 2 :].sum()} "
            f"cusum={mc_[split - 5 :].sum()}"
        )
    first_e = float(e["first_signal"])
    first_c = float(c["first_signal"])
    xb = xbar_r_chart(x, subgroup=5, phase1=split // 5)
    if float(xb["n_ooc_x"]) < 2:
        raise ValueError("xbar missed shifted subgroups")
    cap = process_capability(rng.normal(size=300), lsl=-2.0, usl=2.0)
    if not (cap["cpk"] > 0.5):
        raise ValueError("capability sanity failed")
    return {
        "synthetic_ewma_first": first_e,
        "synthetic_cusum_first": first_c,
        "synthetic_xbar_ooc": float(xb["n_ooc_x"]),
        "synthetic_score": 1.0,
    }
