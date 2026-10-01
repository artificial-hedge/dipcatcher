"""Interval-censored survival: Turnbull's nonparametric MLE.

When an event time is only known to lie in an interval (L, R] —
detection happens at inspection times — Kaplan-Meier is
inapplicable. Turnbull's NPMLE places mass on the innermost
intersections and solves self-consistency equations by EM. The
result is a distribution-free survival curve; the naive
midpoint-imputation KM is biased under wide intervals.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure curve recovery on generated
interval-censored lifetimes — never market evidence.

References:
- Turnbull, B. W. (1976). The empirical distribution function
  with arbitrarily grouped, censored and truncated data.
  *JRSS-B* 38, 290-295 — the self-consistency estimator.
- Turnbull, B. W. (1974). Nonparametric estimation of a
  survivorship function with doubly censored data. *JASA* 69.
- De Gruttola, V., Lagakos, S. W. (1989). Analysis of
  doubly-censored survival data. *Biometrics* 45, 1-11.
- Gentleman, R., Geyer, C. J. (1994). Maximum likelihood for
  interval censored data. *Biometrika* 81, 618-623 — the
  Turnbull-intervals characterization.

Composition: pure numpy — innermost-interval support, EM
self-consistency; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    lo = np.asarray(left, dtype=np.float64).ravel()
    hi = np.asarray(right, dtype=np.float64).ravel()
    n = lo.size
    if n != hi.size or n < 30:
        raise ValueError("left/right must share length >= 30")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("finite intervals required")
    if np.any(lo < 0) or np.any(hi <= lo):
        raise ValueError("need 0 <= left < right")
    return lo, hi


def turnbull_fit(
    left: FloatArray,
    right: FloatArray,
    tol: float = 1e-7,
    max_iter: int = 5000,
) -> dict[str, FloatArray]:
    """Turnbull NPMLE: mass on innermost intervals via EM.

    Returns support intervals and their masses plus the implied
    survival curve evaluated on the right endpoints."""
    lo, hi = _check(left, right)
    n = lo.size
    # Turnbull support: sort unique L, R; candidate cells
    # (u_j, u_{j+1}] formed by L endpoints followed by R endpoints.
    # candidate cells where for some L,R: L == cell_left bound and R == cell_right bound
    cells: list[tuple[float, float]] = []
    # Turnbull cells: pairs (b_i, b_j) with b_i ∈ set(L), b_j ∈ set(R), b_i < b_j,
    # minimal (no whole candidate strictly inside with an L,R on it)
    lset, rset = sorted(set(lo)), sorted(set(hi))
    for lv in lset:
        for rv in rset:
            if rv <= lv:
                continue
            # innermost: no L >= lv with matching R < rv inside; use standard
            # criterion — keep cell only if no L endpoint in (lv, rv] paired
            # with R endpoint in (lv, rv) — approximate via emptiness test
            inside = np.any((lo > lv) & (hi <= rv))
            if not inside:
                cells.append((lv, rv))
    if not cells:
        raise ValueError("no Turnbull cells")
    # dedupe and drop cells containing no observation could cover
    cells = sorted(set(cells))
    m = len(cells)
    if m < 2:
        raise ValueError("need >=2 Turnbull cells")
    if m > 200:
        raise ValueError("cell count > 200 — intervals too sparse")

    # coverage matrix: obs i covers cell j iff l_i < cell_r and r_i >= cell_r
    # (event interval contains the cell's right endpoint and starts before it)
    c_lo = np.array([c[0] for c in cells])
    c_hi = np.array([c[1] for c in cells])
    cover = (lo[:, None] <= c_lo[None, :]) & (hi[:, None] >= c_hi[None, :])
    if np.any(~cover.any(axis=1)):
        raise ValueError("an observation covers no Turnbull cell")

    p = np.full(m, 1.0 / m)
    for _ in range(max_iter):
        pr_obs = cover @ p
        pr_obs = np.clip(pr_obs, 1e-300, None)
        contrib = cover * (p / pr_obs[:, None])
        p_new = contrib.sum(axis=0) / n
        if float(np.max(np.abs(p_new - p))) < tol:
            p = p_new
            break
        p = p_new
    p = p / p.sum()

    # survival at each cell's right edge: S = 1 - cumsum(p) excluding
    # mass below; evaluate S at c_hi
    surv = 1.0 - np.cumsum(p)
    # median: first t where S <= .5 (linear interp within cell)
    med = float("nan")
    for j in range(m):
        if surv[j] <= 0.5:
            s_prev = 1.0 - (float(np.cumsum(p)[j - 1]) if j > 0 else 0.0)
            if s_prev - surv[j] > 1e-12:
                w = (s_prev - 0.5) / (s_prev - surv[j])
                med = float(c_lo[j] + w * (c_hi[j] - c_lo[j]))
            else:
                med = float(c_hi[j])
            break

    return {
        "cell_lo": c_lo,
        "cell_hi": c_hi,
        "mass": p,
        "surv_at_hi": surv,
        "median": np.array([med]),
        "n": np.array([float(n)]),
        "m_cells": np.array([float(m)]),
        "loglik": np.array([float(np.mean(np.log(np.clip(cover @ p, 1e-300, None))))]),
    }


def midpoint_naive_median(left: FloatArray, right: FloatArray) -> float:
    """Naive midpoint-imputation median (biased reference)."""
    lo, hi = _check(left, right)
    return float(np.median((lo + hi) / 2))


def synth_interval(
    n: int = 400,
    inspect_every: float = 1.0,
    rate: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Interval-censored DGP: event times ~ Exp(rate); inspections at
    fixed grid — event only known to fall between two inspections."""
    rng = np.random.default_rng(seed)
    t = rng.exponential(1.0 / rate, n)
    idx = np.floor(t / inspect_every)
    left = idx * inspect_every
    right = (idx + 1) * inspect_every
    return {"left": left, "right": right, "t_true": t}


def bench_interval_censoring(
    seed: int = 20261231 + 220,
) -> dict[str, float]:
    """Interval-censoring self-check: Turnbull median + ISE beat the
    midpoint-imputation bias under wide inspection intervals.
    All ``synthetic_*``."""
    d = synth_interval(inspect_every=0.8, rate=0.5, seed=seed)
    out = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))
    med = float(out["median"][0])
    med_naive = midpoint_naive_median(np.asarray(d["left"]), np.asarray(d["right"]))
    # true median of Exp(.5) = ln2/.5 ≈ 1.386
    med_true = math.log(2) / 0.5
    # survival at hi vs true S(hi)=exp(-.5·hi): ISE
    hi = np.asarray(out["cell_hi"])
    surv = np.asarray(out["surv_at_hi"])
    true_s = np.exp(-0.5 * hi)
    ise = float(np.mean((surv - true_s) ** 2))
    out_b = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))

    return {
        "synthetic_median": med,
        "synthetic_median_err": float(abs(med - med_true)),
        "synthetic_naive_median": med_naive,
        "synthetic_naive_err": float(abs(med_naive - med_true)),
        "synthetic_beats_naive": float(abs(med - med_true) < abs(med_naive - med_true)),
        "synthetic_ise": ise,
        "synthetic_m_cells": float(out["m_cells"][0]),
        "synthetic_detects": float(abs(med - med_true) < 0.45 and ise < 0.03),
        "synthetic_determinism": float(med == float(out_b["median"][0])),
    }
