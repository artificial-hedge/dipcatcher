"""Callaway–Sant'Anna group-time average treatment effects (ATT(g,t)).

Difference-in-differences with multiple periods and staggered treatment
adoption: the treatment effect is decomposed by cohort ``g`` (first
treated period) and calendar period ``t >= g``::

    ATT(g,t) = E[ Y_t - Y_{g-1} | G = g ]  -  E[ Y_t - Y_{g-1} | C ]

where ``C`` is the never-treated group (``control="never"``) or the
not-yet-treated at ``t`` (``control="notyet"``). Estimation here is the
*outcome-regression* identification of CS(2021, thm. 1): the cohort
change is mean-adjusted by the control change — unconditionally
parallel, i.e. the plain difference of long differences, which is the
DR estimator under no covariates. Standard errors come from a
nonparametric cluster bootstrap over units (resample unit rows with
replacement, refit all ATTs); a Wald pretrend test asks whether the
pre-period pseudo-ATTs are jointly zero.

References
----------
- Callaway & Sant'Anna (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics* 225(2):200–230 —
  arXiv:1803.09015.
- Sant'Anna & Zhao (2020). Doubly robust difference-in-differences
  estimators. *Journal of Econometrics* 219(1):101–122 —
  arXiv:1812.01723.

Honesty
-------
All evaluations are SYNTHETIC staggered-adoption panels. Reported keys
are bias/coverage diagnostics of the estimator, never treatment effects
on real portfolios or market outcomes.

Composition notes
-----------------
- ``models/causal_panel.py``: classic two-period DiD and TWFE
  regression. This module implements the CS decomposition that fixes
  TWFE's negative-weighting problem under staggered adoption +
  heterogeneous effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

NEVER_TREATED = 0  # sentinel: cohort code for never-treated units


@dataclass(frozen=True)
class AttGt:
    """Group-time ATT table.

    ``g`` cohort (first treated period), ``t`` evaluation period,
    ``att`` estimate, ``se`` analytic per-cell standard error
    (s_g^2/n_g + s_c^2/n_c on the long differences), ``n`` cohort size,
    ``pretrend`` marks periods t < g (pseudo-effects).
    """

    g: IntArray
    t: IntArray
    att: FloatArray
    se: FloatArray
    n: IntArray
    pretrend: NDArray[np.bool_]


@dataclass(frozen=True)
class AggResult:
    """Aggregated ATT: per-cell ``value`` and the group labels."""

    kind: str
    keys: IntArray
    value: FloatArray
    weights: FloatArray


def _as_panel(data: dict[str, FloatArray]) -> tuple[FloatArray, IntArray, IntArray, FloatArray]:
    """Return (unit, g, t, y) as aligned 1-D arrays of rows."""
    for k in ("unit", "g", "t", "y"):
        if k not in data:
            raise ValueError(f"data missing column {k!r}")
    unit = np.asarray(data["unit"], dtype=np.float64).ravel()
    g = np.asarray(data["g"], dtype=np.intp).ravel()
    t = np.asarray(data["t"], dtype=np.intp).ravel()
    y = np.asarray(data["y"], dtype=np.float64).ravel()
    if not (unit.size == g.size == t.size == y.size) or y.size == 0:
        raise ValueError("columns must be non-empty and equal length")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    return unit, g, t, y


def att_gt(
    data: dict[str, FloatArray],
    control: str = "never",
    periods: int | None = None,
) -> AttGt:
    """Estimate ATT(g,t) for every treated cohort g and period t.

    ``data`` carries columns ``unit`` (unit id), ``g`` (first treated
    period; 0 = never treated), ``t`` (period), ``y`` (outcome).
    ``control`` chooses the comparison group: ``'never'`` (never-treated
    units only) or ``'notyet'`` (never-treated plus cohorts first treated
    after ``t``). Unconditional parallel trends (no covariates): the
    estimator is the difference of long differences.
    """
    if control not in ("never", "notyet"):
        raise ValueError("control must be 'never' or 'notyet'")
    unit, g, t, y = _as_panel(data)
    periods_all = np.unique(t)
    if periods is None:
        periods = int(periods_all.size)
    elif periods != periods_all.size:
        raise ValueError("periods mismatch with data")
    cohorts = np.unique(g[g > 0])
    if cohorts.size == 0:
        raise ValueError("no treated cohorts")
    never = g == NEVER_TREATED
    if control == "never" and not never.any():
        raise ValueError("control='never' but no never-treated units")
    # long difference Δ_i(t, g-1) = y_i(t) - y_i(g-1), per unit
    g_out: list[int] = []
    t_out: list[int] = []
    att_out: list[float] = []
    se_out: list[float] = []
    n_out: list[int] = []
    pre_out: list[bool] = []
    base_mask = {int(p): t == p for p in periods_all}
    for gg in cohorts:
        g_mask = g == gg
        n_g = int(g_mask.sum())
        base = int(gg) - 1
        if base not in base_mask:
            continue
        base_rows = base_mask[base]
        # restrict to units present at base and at t
        for tt in periods_all:
            rows_t = t == tt
            dy_g = y[g_mask & rows_t] - y[g_mask & base_rows]
            if control == "never":
                c_mask = never
            else:
                c_mask = never | (g > tt)
            dy_c = y[c_mask & rows_t] - y[c_mask & base_rows]
            if dy_g.size == 0 or dy_c.size == 0:
                continue
            var_g = float(np.var(dy_g, ddof=1)) if dy_g.size > 1 else 0.0
            var_c = float(np.var(dy_c, ddof=1)) if dy_c.size > 1 else 0.0
            var = var_g / dy_g.size + var_c / dy_c.size
            g_out.append(int(gg))
            t_out.append(int(tt))
            att_out.append(float(dy_g.mean() - dy_c.mean()))
            se_out.append(math.sqrt(max(var, 1e-12)))
            n_out.append(n_g)
            pre_out.append(bool(tt < gg))
    return AttGt(
        g=np.asarray(g_out, dtype=np.intp),
        t=np.asarray(t_out, dtype=np.intp),
        att=np.asarray(att_out, dtype=np.float64),
        se=np.asarray(se_out, dtype=np.float64),
        n=np.asarray(n_out, dtype=np.intp),
        pretrend=np.asarray(pre_out, dtype=np.bool_),
    )


def agg_att(res: AttGt, kind: str = "simple") -> AggResult:
    """Aggregate group-time ATTs.

    ``kind='simple'`` — cohort-size-weighted mean of post-treatment
    cells. ``'calendar'`` — mean by calendar t. ``'event'`` — mean by
    event time e = t − g. ``'group'`` — mean by cohort g.
    """
    if kind not in ("simple", "calendar", "event", "group"):
        raise ValueError("kind must be simple|calendar|event|group")
    post = ~res.pretrend
    if not post.any():
        raise ValueError("no post-treatment cells")
    g, t, att, n = res.g[post], res.t[post], res.att[post], res.n[post]
    if kind == "simple":
        w = n.astype(np.float64)
        return AggResult(
            kind=kind,
            keys=np.array([0], dtype=np.intp),
            value=np.array([float((att * w).sum() / w.sum())]),
            weights=np.array([w.sum()]),
        )
    labels = {"calendar": t, "event": t - g, "group": g}[kind]
    uniq = np.unique(labels)
    val = np.empty(uniq.size)
    wts = np.empty(uniq.size)
    for i, lab in enumerate(uniq):
        m = labels == lab
        w = n[m].astype(np.float64)
        val[i] = float((att[m] * w).sum() / w.sum())
        wts[i] = float(w.sum())
    return AggResult(kind=kind, keys=uniq.astype(np.intp), value=val, weights=wts)


def boot_se(
    data: dict[str, FloatArray],
    control: str = "never",
    kind: str = "simple",
    n_boot: int = 100,
    seed: int = 0,
) -> FloatArray:
    """Cluster bootstrap SEs of the aggregated ATT (resample units)."""
    unit, g, t, y = _as_panel(data)
    units = np.unique(unit)
    rng = np.random.default_rng(seed)
    boots: list[FloatArray] = []
    for _b in range(n_boot):
        take = rng.choice(units, size=units.size, replace=True)
        idx = np.concatenate([np.flatnonzero(unit == u) for u in take])
        d = {
            "unit": unit[idx],
            "g": g[idx],
            "t": t[idx],
            "y": y[idx],
        }
        try:
            res = att_gt(d, control=control)
            agg = agg_att(res, kind=kind)
            boots.append(agg.value)
        except ValueError:
            continue
    if not boots:
        raise ValueError("no successful bootstrap replicates")
    return np.stack(boots).std(axis=0, ddof=1)


def pretrend_test(res: AttGt) -> tuple[float, float]:
    """Wald test that pre-period pseudo-ATTs are jointly zero.

    Returns (chi2_stat, p_value). Uses the diagonal approximation
    (per-cell variance from the cohort size scaling) — adequate for the
    SYNTHETIC pretrend gate; not a substitute for bootstrap inference.
    """
    pre = res.pretrend
    if not pre.any():
        raise ValueError("no pre-period cells")
    att_pre = res.att[pre]
    v = np.maximum(res.se[pre] ** 2, 1e-12)
    chi2 = float(np.sum(att_pre**2 / v))
    dof = att_pre.size
    p = float(sstats.chi2.sf(chi2, dof))
    return chi2, p


def synth_panel(
    n_units: int = 200,
    periods: int = 8,
    treat_periods: tuple[int, ...] = (4, 6),
    treat_frac: tuple[float, float] = (0.3, 0.3),
    tau_g: float = 2.0,
    tau_growth: float = 0.5,
    pretrend_slope: float = 0.0,
    noise: float = 0.5,
    seed: int = 0,
) -> tuple[dict[str, FloatArray], dict[int, float]]:
    """Staggered-adoption panel.

    Units split into cohorts ``treat_periods`` (sizes ``treat_frac``)
    plus never-treated. True effect in cohort g at event time e is
    ``tau_g + tau_growth * e``; optional ``pretrend_slope`` adds a
    cohort-specific linear trend pre-treatment (breaks parallel trends).
    Returns (data dict, true_effects {g: mean post-treatment ATT}).
    """
    if n_units < 20 or periods < 3 or len(treat_periods) != len(treat_frac):
        raise ValueError("invalid panel spec")
    rng = np.random.default_rng(seed)
    n_treat = [int(n_units * f) for f in treat_frac]
    if sum(n_treat) >= n_units:
        raise ValueError("treat_frac too large")
    g_assign = np.zeros(n_units, dtype=np.intp)
    ofs = 0
    for tp, nn in zip(treat_periods, n_treat, strict=True):
        g_assign[ofs : ofs + nn] = tp
        ofs += nn
    rng.shuffle(g_assign)
    unit_eff = rng.standard_normal(n_units) * 1.0
    time_eff = np.arange(periods) * 0.3 + rng.standard_normal(periods) * 0.1
    rows_u: list[float] = []
    rows_g: list[float] = []
    rows_t: list[float] = []
    rows_y: list[float] = []
    true_att: dict[int, float] = {}
    for g_cohort in treat_periods:
        post_e = np.arange(g_cohort, periods) - g_cohort
        true_att[g_cohort] = float((tau_g + tau_growth * post_e).mean())
    for i in range(n_units):
        gc = int(g_assign[i])
        for tp in range(periods):
            yv = unit_eff[i] + time_eff[tp] + rng.standard_normal() * noise
            if gc > 0:
                e = tp - gc
                if e >= 0:
                    yv += tau_g + tau_growth * e
                elif pretrend_slope != 0.0:
                    yv += pretrend_slope * e  # linear pretrend (e<0)
            rows_u.append(float(i))
            rows_g.append(float(gc))
            rows_t.append(float(tp))
            rows_y.append(yv)
    data = {
        "unit": np.asarray(rows_u),
        "g": np.asarray(rows_g),
        "t": np.asarray(rows_t),
        "y": np.asarray(rows_y),
    }
    return data, true_att


def bench_callaway_did(seed: int = 20261231 + 154) -> dict[str, float]:
    """SYNTHETIC CS-DiD telemetry."""
    data, true_att = synth_panel(n_units=300, periods=8, seed=seed)
    res = att_gt(data, control="never")
    agg = agg_att(res, "simple")
    true_simple = float(np.mean(list(true_att.values())))
    bias = float(abs(agg.value[0] - true_simple))
    # event aggregation bias: synth effect is tau_g + growth*e for every
    # cohort, so the truth at event time e is 2.0 + 0.5*e
    agg_e = agg_att(res, "event")
    e_bias = float(
        np.mean(
            [abs(v - (2.0 + 0.5 * int(k))) for k, v in zip(agg_e.keys, agg_e.value, strict=True)]
        )
    )
    # pretrend: clean panel gives large p
    _chi, p_clean = pretrend_test(res)
    data_pre, _ = synth_panel(n_units=300, periods=8, pretrend_slope=1.2, seed=seed + 1)
    _chi2, p_pre = pretrend_test(att_gt(data_pre, control="never"))
    # coverage: fraction of post-treatment cells whose 95% analytic CI
    # brackets the true per-cell effect tau_g + growth*(t-g)
    post = ~res.pretrend
    truth_cell = 2.0 + 0.5 * (res.t[post] - res.g[post])
    covers = float(np.mean(np.abs(res.att[post] - truth_cell) <= 1.96 * res.se[post] + 1e-9))
    # bootstrap SE of the simple aggregate (reported, sanity > 0)
    se = boot_se(data, control="never", kind="simple", n_boot=40, seed=seed)
    # never vs notyet agreement
    res_ny = att_gt(data, control="notyet")
    agree = float(abs(agg_att(res_ny, "simple").value[0] - agg.value[0]))
    # determinism
    res2 = att_gt(data, control="never")
    det = float(np.array_equal(res.att, res2.att))
    return {
        "synthetic_att_bias": bias,
        "synthetic_att_event_bias": e_bias,
        "synthetic_pretrend_p": p_clean,
        "synthetic_pretrend_reject_p": p_pre,
        "synthetic_ci_covers": covers,
        "synthetic_boot_se": float(se[0]),
        "synthetic_never_vs_notyet_absdiff": agree,
        "synthetic_n_att_cells": float(res.att.size),
        "synthetic_determinism": det,
    }
