"""DiD diagnostics: Goodman-Bacon decomposition + Sun-Abraham
event-study aggregation.

Goodman-Bacon: a two-way fixed-effects DiD coefficient is a weighted
average of ALL possible 2x2 DiD comparisons across timing groups —
including "forbidden" comparisons that use already-treated units as
controls (which bias the estimate under effect heterogeneity). This
module computes each 2x2 component and its weight, plus the
Sun-Abraham cohort-aggregation diagnostic (event-study CATTs combined
without forbidden controls).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure diagnostic recovery on generated
staggered-adoption panels — never market evidence.

References:
- Goodman-Bacon (2021). Difference-in-differences with variation in
  treatment timing. *Journal of Econometrics* 225.
- Sun, Abraham (2021). Estimating dynamic treatment effects in event
  studies with heterogeneous treatment effects. *J. Econometrics* 225.
- Callaway, Sant'Anna (2021). Difference-in-differences with multiple
  time periods. *J. Econometrics* 225.
- de Chaisemartin, D'Haultfoeuille (2020). Two-way fixed effects
  estimators with heterogeneous treatment effects. *AER* 110.

Composition: pure numpy — exhaustive 2x2 enumeration on collapsed
group-time means, variance-share weights, cohort event-study CATTs;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_panels(
    y: FloatArray, unit: FloatArray, time: FloatArray, g: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    ya = np.asarray(y, dtype=np.float64).ravel()
    ua = np.asarray(unit).ravel()
    ta = np.asarray(time).ravel()
    ga = np.asarray(g, dtype=np.float64).ravel()  # first-treated period; 0=never
    n = ya.size
    if ua.size != n or ta.size != n or ga.size != n:
        raise ValueError("y/unit/time/g length mismatch")
    if not np.all(np.isfinite(ya)):
        raise ValueError("y must be finite")
    return ya, ua, ta, ga


def twfe_beta(y: FloatArray, unit: FloatArray, time: FloatArray, g: FloatArray) -> float:
    """TWFE DiD coefficient: residualize on unit + time dummies, regress
    on post-treatment dummy."""
    ya, ua, ta, ga = _as_panels(y, unit, time, g)
    units = np.unique(ua)
    times = np.unique(ta)
    U = np.zeros((n := ya.size, units.size))
    T = np.zeros((n, times.size))
    for i, u in enumerate(units):
        U[ua == u, i] = 1.0
    for j, t in enumerate(times):
        T[ta == t, j] = 1.0
    post_treat = ((ta >= ga) & (ga > 0)).astype(np.float64)
    Z = np.column_stack([post_treat, U[:, 1:], T[:, 1:]])
    beta, *_ = np.linalg.lstsq(Z, ya, rcond=None)
    return float(beta[0])


def bacon_decomposition(
    y: FloatArray, unit: FloatArray, time: FloatArray, g: FloatArray
) -> dict[str, float | FloatArray]:
    """Exhaustive 2x2 decomposition of the TWFE DiD coefficient.

    For each pair of timing groups (g_i < g_j or g_i vs never-treated),
    and each timing window, compute the 2x2 DiD and its weight
    (variance share). Returns components and the clean-vs-forbidden
    weight split — the share of TWFE variation comparing
    later-treated vs already-treated groups.
    """
    ya, ua, ta, ga = _as_panels(y, unit, time, g)
    times = np.unique(ta)
    groups = np.unique(ga[ga > 0])
    if groups.size < 1:
        raise ValueError("need at least one treated cohort")
    never = ga <= 0
    if not never.any():
        raise ValueError("need a never-treated comparison group")

    # collapse to group x time means
    all_g = np.unique(ga)
    gm = np.empty((all_g.size, times.size))
    gn = np.empty((all_g.size, times.size))
    for i, gv in enumerate(all_g):
        for j, tv in enumerate(times):
            sel = (ga == gv) & (ta == tv)
            gm[i, j] = float(ya[sel].mean()) if sel.any() else np.nan
            gn[i, j] = float(sel.sum())
    if np.isnan(gm).any():
        raise ValueError("unbalanced group-time cells")

    comp_did: list[float] = []
    comp_w: list[float] = []
    comp_kind: list[
        float
    ] = []  # 1 = clean (vs never / vs not-yet), 0 = forbidden (vs already-treated)

    def did2x2(y_pre_t: float, y_post_t: float, y_pre_c: float, y_post_c: float) -> float:
        return (y_post_t - y_pre_t) - (y_post_c - y_pre_c)

    n_times = times.size
    n_tot = float(np.sum(gn))

    def gb_w(i: int, j: int, lo: int, hi: int) -> float:
        # exact GB variance weight: subsample = groups i,j x periods [lo,hi)
        # split at the earlier treatment; p = share of (i, later-half) cells.
        n_ij = float(gn[i].sum() + gn[j].sum()) / n_tot
        t_span = float(hi - lo)
        mid = lo + (hi - lo) // 2
        p = float((hi - mid) * float(gn[i].sum())) / max(
            float(t_span) * float(gn[i].sum() + gn[j].sum()), 1.0
        )
        return n_ij * t_span * p * (1.0 - p)

    for i, gi in enumerate(all_g):
        if gi <= 0:
            continue
        gi_idx = int(np.flatnonzero(times == gi)[0])
        # vs never-treated: DiD across pre/post split at gi
        j0 = int(np.flatnonzero(all_g <= 0)[0])
        pre = float(np.mean(gm[i, :gi_idx])) - float(np.mean(gm[j0, :gi_idx]))
        post = float(np.mean(gm[i, gi_idx:])) - float(np.mean(gm[j0, gi_idx:]))
        comp_did.append(post - pre)
        comp_w.append(gb_w(i, j0, 0, n_times))
        comp_kind.append(1.0)
        # vs other treated cohorts
        for j, gj in enumerate(all_g):
            if gj <= 0 or gj == gi:
                continue
            if gj > gi:
                # i early-treated vs j later-treated (not yet treated
                # in [gi, gj)): clean window
                lo, hi = gi_idx, int(np.flatnonzero(times == gj)[0])
                if hi - lo >= 1:
                    d = did2x2(
                        float(np.mean(gm[i, :lo])),
                        float(np.mean(gm[i, lo:hi])),
                        float(np.mean(gm[j, :lo])),
                        float(np.mean(gm[j, lo:hi])),
                    )
                    comp_did.append(d)
                    comp_w.append(gb_w(i, j, 0, hi))
                    comp_kind.append(1.0)
                # late window: i treated, j treated after — both treated → forbidden
                if n_times - hi >= 1 and lo >= 1:
                    d = did2x2(
                        float(np.mean(gm[i, lo:hi])),
                        float(np.mean(gm[i, hi:])),
                        float(np.mean(gm[j, lo:hi])),
                        float(np.mean(gm[j, hi:])),
                    )
                    comp_did.append(d)
                    comp_w.append(gb_w(i, j, lo, n_times))
                    comp_kind.append(0.0)
            else:
                # j earlier-treated than i: i's window [gi,T] post,
                # j already treated → forbidden
                gj_idx = int(np.flatnonzero(times == gj)[0])
                lo, hi = gj_idx, gi_idx
                if hi - lo >= 1 and n_times - hi >= 1:
                    d = did2x2(
                        float(np.mean(gm[i, lo:hi])),
                        float(np.mean(gm[i, hi:])),
                        float(np.mean(gm[j, lo:hi])),
                        float(np.mean(gm[j, hi:])),
                    )
                    comp_did.append(d)
                    comp_w.append(gb_w(i, j, lo, n_times))
                    comp_kind.append(0.0)

    dd = np.asarray(comp_did)
    ww = np.asarray(comp_w)
    kk = np.asarray(comp_kind)
    w_norm = ww / ww.sum()
    beta_bacon = float(np.sum(dd * w_norm))
    twfe = twfe_beta(y, unit, time, g)
    clean_w = float(w_norm[kk > 0.5].sum()) if (kk > 0.5).any() else 0.0
    return {
        "beta_twfe": twfe,
        "beta_bacon": beta_bacon,
        "bacon_gap": abs(beta_bacon - twfe),
        "clean_weight": clean_w,
        "forbidden_weight": 1.0 - clean_w,
        "n_components": float(dd.size),
        "components": dd,
        "weights": w_norm,
        "kinds": kk,
    }


def sun_abraham_effects(
    y: FloatArray,
    unit: FloatArray,
    time: FloatArray,
    g: FloatArray,
    rel_periods: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> dict[str, FloatArray | float]:
    """Cohort-specific event-study CATTs at relative periods, never-treated
    controls only (Sun-Abraham aggregation across cohorts weighted by
    cohort size)."""
    ya, ua, ta, ga = _as_panels(y, unit, time, g)
    times = np.unique(ta)
    groups = np.unique(ga[ga > 0])
    never = ga <= 0
    if not never.any() or groups.size < 1:
        raise ValueError("need treated cohorts and never-treated controls")

    effs: list[float] = []
    wts: list[float] = []
    for rp in rel_periods:
        num, wsum = 0.0, 0.0
        for gv in groups:
            t_eval = gv + rp
            if t_eval not in times or gv - 1 not in times:
                continue
            sel_t = (ga == gv) & (ta == t_eval)
            sel_t0 = (ga == gv) & (ta == gv - 1)
            sel_c = (ga <= 0) & (ta == t_eval)
            sel_c0 = (ga <= 0) & (ta == gv - 1)
            if not (sel_t.any() and sel_c.any()):
                continue
            catt = (float(ya[sel_t].mean()) - float(ya[sel_t0].mean())) - (
                float(ya[sel_c].mean()) - float(ya[sel_c0].mean())
            )
            w = float(sel_t.sum())
            num += w * catt
            wsum += w
        if wsum > 0:
            effs.append(num / wsum)
            wts.append(float(rp))
    return {
        "rel_periods": np.asarray(wts),
        "catt": np.asarray(effs),
        "mean_post_catt": float(np.mean([e for e, rp in zip(effs, wts, strict=True) if rp >= 0]))
        if any(rp >= 0 for rp in wts)
        else math.nan,
    }


def synth_staggered(
    n_per_group: int = 40,
    t: int = 24,
    cohorts: tuple[int, ...] = (12, 18),
    tau_step: float = 0.6,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Staggered-adoption panel: cohort effects grow with time-since-
    adoption (τ_k = tau_step·k) so TWFE is downward-biased vs CATT and
    the forbidden weight is material — textbook GB heterogeneity."""
    rng = np.random.default_rng(seed)
    n_cohort = n_per_group * (len(cohorts) + 1)
    units = np.arange(n_cohort)
    gvals = np.concatenate([np.full(n_per_group, c) for c in cohorts] + [np.zeros(n_per_group)])
    rng.shuffle(gvals)
    yl, ul, tl, gl = [], [], [], []
    for i, u in enumerate(units):
        gv = gvals[i]
        fe_u = rng.normal(0, 0.3)
        for tt in range(t):
            fe_t = 0.05 * tt
            treated_post = gv > 0 and tt >= gv
            tau = tau_step * (tt - gv + 1) if treated_post else 0.0
            y = fe_u + fe_t + tau + rng.normal(0, 0.2)
            yl.append(y)
            ul.append(u)
            tl.append(tt)
            gl.append(gv)
    return {
        "y": np.asarray(yl),
        "unit": np.asarray(ul),
        "time": np.asarray(tl),
        "g": np.asarray(gl),
        "tau_step": np.full(1, tau_step),
        "cohorts": np.asarray(cohorts),
    }


def bench_did_diagnostics(seed: int = 20261231 + 196) -> dict[str, float]:
    """GB/SA self-check: under growing-with-time treatment effects the
    TWFE is biased down vs the Sun-Abraham post CATT; Bacon weights
    recover the coefficient; forbidden share > 0. All ``synthetic_*``."""
    d = synth_staggered(seed=seed, tau_step=0.6)
    y = np.asarray(d["y"])
    u = np.asarray(d["unit"])
    t = np.asarray(d["time"])
    g = np.asarray(d["g"])

    bd = bacon_decomposition(y, u, t, g)
    sa = sun_abraham_effects(y, u, t, g)
    mean_catt = float(sa["mean_post_catt"])
    twfe = float(bd["beta_twfe"])
    bacon = float(bd["beta_bacon"])

    d0 = synth_staggered(seed=seed + 1, tau_step=0.0)
    bd0 = bacon_decomposition(
        np.asarray(d0["y"]), np.asarray(d0["unit"]), np.asarray(d0["time"]), np.asarray(d0["g"])
    )

    return {
        "synthetic_beta_twfe": twfe,
        "synthetic_beta_bacon": bacon,
        "synthetic_bacon_gap": float(bd["bacon_gap"]),
        "synthetic_forbidden_w": float(bd["forbidden_weight"]),
        "synthetic_sa_mean_catt": mean_catt,
        "synthetic_twfe_bias": abs(twfe - mean_catt),
        "synthetic_bacon_gap_small": float(bd["bacon_gap"] < 0.6),
        "synthetic_null_twfe": float(bd0["beta_twfe"]),
        "synthetic_detects": float(mean_catt > 0.5 and abs(twfe - mean_catt) > 0.5),
        "synthetic_determinism": float(
            twfe == float(bacon_decomposition(np.asarray(d["y"]), u, t, g)["beta_twfe"])
        ),
    }
