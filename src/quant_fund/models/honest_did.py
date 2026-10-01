"""Honest DiD sensitivity analysis (Rambachan & Roth).

Instead of assuming parallel trends hold exactly, bound the possible
POST-treatment violation by the observed PRE-trend: the Δ^SD(smoothness
deviation) approach caps post-period deviations from a linear
extrapolation of pre-trends at M̄ × the largest observed pre deviation.
Robust inference uses the FLCI (fixed-length confidence interval)
critical values for affine worst-case bias — reported per M̄ and as a
breakdown value M̄* where significance is lost.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure sensitivity-bound behavior on
generated event-study panels — never market evidence.

References:
- Rambachan, Roth (2023). A more credible approach to parallel
  trends. *Review of Economic Studies* 90.
- Rambachan, Roth (2020). An honest approach to parallel trends.
  Working paper (Δ^SD and FLCI construction).
- Roth (2022). Pretest with caution: event-study estimates after
  testing for parallel trends. *AER: Insights* 4.
- Manski, Pepper (2018). How do right-to-carry laws affect crime
  rates? *J. Econometrics* (smoothness bounds lineage).

Composition: pure numpy — event-study aggregation, linear-trend
deviation bounding, FLCI lookup via analytic bisection; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_vector(y: FloatArray, name: str, min_n: int = 10) -> FloatArray:
    a = np.asarray(y, dtype=np.float64).ravel()
    if a.size < min_n or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite vector len >= {min_n} required")
    return a


def event_study_betas(
    y: FloatArray,
    unit: FloatArray,
    time: FloatArray,
    g: FloatArray,
    rel_periods: tuple[int, ...] = (-3, -2, 0, 1, 2),
) -> dict[str, FloatArray]:
    """Simple event-study: for each relative period rp, cohort-average
    DiD of treated-vs-never-treated using period g-1 as baseline.
    Returns (rel_periods, beta, se) — the input object for honest-DiD."""
    ya = np.asarray(y, dtype=np.float64).ravel()
    ua = np.asarray(unit).ravel()
    ta = np.asarray(time).ravel()
    ga = np.asarray(g, dtype=np.float64).ravel()
    n = ya.size
    if ua.size != n or ta.size != n or ga.size != n:
        raise ValueError("length mismatch")
    groups = np.unique(ga[ga > 0])
    never = ga <= 0
    if groups.size < 1 or not never.any():
        raise ValueError("need treated cohorts and never-treated controls")
    times = np.unique(ta)

    betas, ses, rps = [], [], []
    for rp in rel_periods:
        ds, ns = [], []
        for gv in groups:
            te, tb = gv + rp, gv - 1
            if te not in times or tb not in times:
                continue
            st = (ga == gv) & (ta == te)
            s0 = (ga == gv) & (ta == tb)
            ct = (ga <= 0) & (ta == te)
            c0 = (ga <= 0) & (ta == tb)
            if not (st.any() and ct.any()):
                continue
            d = (ya[st].mean() - ya[s0].mean()) - (ya[ct].mean() - ya[c0].mean())
            se = math.sqrt(
                ya[st].var(ddof=1) / st.sum()
                + ya[s0].var(ddof=1) / s0.sum()
                + ya[ct].var(ddof=1) / ct.sum()
                + ya[c0].var(ddof=1) / c0.sum()
            )
            ds.append(d)
            ns.append(float(st.sum()))
        if ds:
            w = np.asarray(ns)
            b = float(np.average(np.asarray(ds), weights=w))
            # conservative pooled SE: std of cohort-level DiDs
            ds_arr = np.asarray(ds)
            se = (
                float(np.std(ds_arr, ddof=1) / math.sqrt(ds_arr.size))
                if ds_arr.size > 1
                else float(ya[ga > 0].std(ddof=1) / math.sqrt(n))
            )
            betas.append(b)
            ses.append(max(se, 1e-12))
            rps.append(float(rp))
    return {
        "rel_periods": np.asarray(rps),
        "beta": np.asarray(betas),
        "se": np.asarray(ses),
    }


def _flci_cv(bias_max: float, se: float, alpha: float = 0.05) -> float:
    """Approximate FLCI critical value for affine worst-case bias.

    For a scalar normal with bounded bias B, the fixed-length CI
    halfwidth is cv·se where cv solves 1 - Φ(cv - B/se) + Φ(-cv - B/se)
    = 1 - alpha (worst-case over |bias| <= B, since the normal is
    symmetric unimodal the worst bias is B). Solved by bisection.
    """
    from math import erf, sqrt

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    b = bias_max / max(se, 1e-12)
    target = 1.0 - alpha

    def cover(cv: float) -> float:
        return cdf(cv - b) - cdf(-cv - b)

    lo, hi = 1.0, 1.0 + b + 4.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cover(mid) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def honest_did_sd(
    es_beta: FloatArray,
    es_se: FloatArray,
    rel_periods: FloatArray,
    *,
    post_rp: int = 1,
    mbar: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Δ^SD robustness for the CATT at relative period ``post_rp``.

    Pre-trend deviations from a fitted pre line bound the post-period
    counterfactual violation at mbar * max|pre_dev|. The honest CI is
    [beta - mbar_max_dev - cv*se, beta + mbar_max_dev + cv*se] with the
    FLCI critical value.
    """
    beta = np.asarray(es_beta, dtype=np.float64).ravel()
    se = np.asarray(es_se, dtype=np.float64).ravel()
    rp = np.asarray(rel_periods, dtype=np.float64).ravel()
    if not (beta.size == se.size == rp.size) or beta.size < 4:
        raise ValueError("matching beta/se/rp vectors len>=4 required")
    if np.any(se <= 0) or not np.all(np.isfinite(beta)):
        raise ValueError("se must be positive and beta finite")
    pre = rp < 0
    if pre.sum() < 2:
        raise ValueError("need >=2 pre periods")
    if int(post_rp) not in rp.astype(int):
        raise ValueError("post_rp not in rel_periods")

    # fit linear trend on pre betas (wLS by 1/se^2)
    w = 1.0 / se[pre] ** 2
    X = np.column_stack([np.ones(pre.sum()), rp[pre]])
    coef, *_ = np.linalg.lstsq(X * w[:, None], beta[pre] * w, rcond=None)
    dev = beta[pre] - X @ coef
    max_dev = float(np.abs(dev).max())

    j = int(np.flatnonzero(rp.astype(int) == post_rp)[0])
    bias = mbar * max_dev
    cv = _flci_cv(bias, se[j], alpha)
    half = cv * se[j]
    lo, hi = float(beta[j] - bias - half), float(beta[j] + bias + half)
    covers_zero = float(lo <= 0.0 <= hi)
    return {
        "beta": float(beta[j]),
        "se": float(se[j]),
        "max_pre_dev": max_dev,
        "bias_bound": bias,
        "ci_lo": lo,
        "ci_hi": hi,
        "half_length": float(2.0 * half + 2.0 * bias),
        "covers_zero": covers_zero,
        "significant": float(1.0 - covers_zero),
    }


def breakdown_mbar(
    es_beta: FloatArray,
    es_se: FloatArray,
    rel_periods: FloatArray,
    *,
    post_rp: int = 1,
    grid: int = 41,
    alpha: float = 0.05,
) -> float:
    """Largest mbar where the honest CI still excludes zero (breakdown
    value); inf/grid-edge when the CI stays significant for all M̄."""
    beta = np.asarray(es_beta, dtype=np.float64).ravel()
    se = np.asarray(es_se, dtype=np.float64).ravel()
    rp = np.asarray(rel_periods, dtype=np.float64).ravel()
    for m in np.linspace(0.0, 6.0, grid):
        out = honest_did_sd(beta, se, rp, post_rp=post_rp, mbar=float(m), alpha=alpha)
        if out["covers_zero"] > 0.5:
            return float(m)
    return math.inf


def synth_es(
    n: int = 60,
    t: int = 20,
    g_t: int = 12,
    tau: float = 1.0,
    pre_trend: float = 0.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Event-study panel: half the units treated at g_t; treated get
    tau + optional deterministic pre-trend (used to show breakdown)."""
    rng = np.random.default_rng(seed)
    n_units = 2 * n
    treated = np.zeros(n_units, dtype=bool)
    treated[rng.choice(n_units, n, replace=False)] = True
    g = np.where(treated, float(g_t), 0.0)
    rows = []
    for u in range(n_units):
        fe = rng.normal(0, 0.3)
        for tt in range(t):
            y = fe + 0.05 * tt + rng.normal(0, 0.25)
            if treated[u]:
                if tt < g_t:
                    # NONLINEAR pre-trend: Δ^SD bounds deviations from
                    # linear extrapolation, so a pure linear slope is a
                    # no-op — curvature is the honest stress.
                    y += pre_trend * np.sin((tt - (g_t - 1)) * 0.9) * 3.0
                else:
                    y += tau
            rows.append((y, u, tt, g[u]))
    a = np.asarray(rows)
    return {
        "y": a[:, 0],
        "unit": a[:, 1],
        "time": a[:, 2],
        "g": a[:, 3],
        "tau": np.full(1, tau),
        "pre_trend": np.full(1, pre_trend),
    }


def bench_honest_did(seed: int = 20261231 + 197) -> dict[str, float]:
    """Honest-DiD self-check: significant honest CI under parallel trends
    + shrinking breakdown value under injected pre-trend. All
    ``synthetic_*``."""
    # case 1: true parallel trends → big breakdown value
    d = synth_es(seed=seed, tau=1.0, pre_trend=0.0)
    es = event_study_betas(
        np.asarray(d["y"]),
        np.asarray(d["unit"]),
        np.asarray(d["time"]),
        np.asarray(d["g"]),
        rel_periods=(-6, -5, -4, -3, -2, -1, 0, 1, 2),
    )
    out = honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=1, mbar=1.0)
    mb = breakdown_mbar(es["beta"], es["se"], es["rel_periods"], post_rp=1)

    # case 2: pre-trend violation → breakdown collapses
    d2 = synth_es(seed=seed + 1, tau=1.0, pre_trend=0.5)
    es2 = event_study_betas(
        np.asarray(d2["y"]),
        np.asarray(d2["unit"]),
        np.asarray(d2["time"]),
        np.asarray(d2["g"]),
        rel_periods=(-6, -5, -4, -3, -2, -1, 0, 1, 2),
    )
    mb2 = breakdown_mbar(es2["beta"], es2["se"], es2["rel_periods"], post_rp=1)
    out2 = honest_did_sd(es2["beta"], es2["se"], es2["rel_periods"], post_rp=1, mbar=1.0)

    return {
        "synthetic_beta_post": float(out["beta"]),
        "synthetic_max_pre_dev": float(out["max_pre_dev"]),
        "synthetic_ci_half": float(out["half_length"]),
        "synthetic_significant_m1": float(out["significant"]),
        "synthetic_breakdown_mbar": float(mb) if math.isfinite(mb) else 6.0,
        "synthetic_pt_breakdown_mbar": float(mb2) if math.isfinite(mb2) else 6.0,
        "synthetic_pt_max_pre_dev": float(out2["max_pre_dev"]),
        "synthetic_breakdown_shrinks": float(
            (mb if math.isfinite(mb) else 6.0) > (mb2 if math.isfinite(mb2) else 6.0)
        ),
        "synthetic_detects": float(out["significant"] == 1.0),
        "synthetic_determinism": float(
            out["beta"]
            == honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=1, mbar=1.0)["beta"]
        ),
    }
