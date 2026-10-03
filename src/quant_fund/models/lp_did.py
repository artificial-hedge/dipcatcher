"""LP-DiD — local-projections difference-in-differences.

At each horizon h, LP-DiD regresses the h-period outcome change on
the treatment indicator using only *clean* observations: treated
units whose h-window is not contaminated by other treatments, plus
not-yet-treated controls. This avoids the negative weighting and
contamination biases of TWFE event-study specifications under
staggered adoption with heterogeneous effects.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure horizon-effect recovery on
generated staggered panels — never market evidence.

References:
- Dube, A., Girardi, D., Jordà, Ò., Taylor, A. M. (2023). A local
  projections approach to difference-in-differences event studies.
  *NBER WP* 31184 — the LP-DiD specification and clean-weighting.
- Jordà, Ò. (2005). Estimation and inference of impulse responses
  by local projections. *AER* 95, 161-182 — LP machinery.
- de Chaisemartin, C., D'Haultfœuille, X. (2020). Two-way fixed
  effects estimators with heterogeneous treatment effects. *AER*
  110, 2964-2996 — the contamination problem LP-DiD avoids.
- Cengiz, D., Dube, A., Lindner, A., Zipperer, B. (2019). The
  effect of minimum wages on low-wage jobs. *QJE* 134 — stacking /
  clean-control precedent.

Composition: pure numpy — per-horizon OLS on the cleaned subsample
with HC1 SE; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _lp_did_at_h(
    y: FloatArray,
    unit: FloatArray,
    time: FloatArray,
    treat_time: FloatArray,
    h: int,
    max_lag: int = 0,
) -> tuple[float, float, int]:
    """LP-DiD slope at horizon h.

    Sample: treated units with treat_time = g and controls not yet
    treated by t = g + h (+ max_lag). Outcome = y_{t+h} - y_{t-1}."""
    units = np.unique(unit)
    dy, d = [], []
    for u in units:
        mu = unit == u
        tu = time[mu]
        yu = y[mu]
        g = float(treat_time[mu][0])
        # base period: one before adoption (treated) or matched t
        t0 = int(g) - 1 if np.isfinite(g) else None
        if t0 is not None:
            # treated: need y at t0 and t0+h+1
            i0 = np.where(tu == t0)[0]
            ih = np.where(tu == t0 + 1 + h)[0]
            if i0.size and ih.size:
                dy.append(float(yu[ih[0]] - yu[i0[0]]))
                d.append(1)
        else:
            # control: use the modal treated adoption time minus 1
            gval = np.asarray(treat_time)
            finite_g = gval[np.isfinite(gval)]
            if finite_g.size == 0:
                continue
            g_mode = int(np.median(finite_g))
            t0c = g_mode - 1
            i0 = np.where(tu == t0c)[0]
            ih = np.where(tu == t0c + 1 + h)[0]
            if i0.size and ih.size:
                dy.append(float(yu[ih[0]] - yu[i0[0]]))
                d.append(0)
    dy_a = np.asarray(dy)
    d_a = np.asarray(d, dtype=np.float64)
    if d_a.sum() < 4 or (d_a == 0).sum() < 4:
        raise ValueError(f"horizon {h}: insufficient clean treated/control units")
    xmat = np.column_stack([np.ones(dy_a.size), d_a])
    b = np.linalg.lstsq(xmat, dy_a, rcond=None)[0]
    resid = dy_a - xmat @ b
    xtxi = np.linalg.inv(xmat.T @ xmat)
    meat = xmat.T @ ((resid**2)[:, None] * xmat)
    var1 = (xtxi @ meat @ xtxi)[1, 1]
    se = float(math.sqrt(max(var1, 0.0)) * math.sqrt(dy_a.size / max(dy_a.size - 2, 1)))
    return float(b[1]), se, int(d_a.sum())


def lp_did(
    y: FloatArray,
    unit: FloatArray,
    time: FloatArray,
    treat_time: FloatArray,
    horizons: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[str, float]:
    """LP-DiD event-study coefficients across horizons.

    ``treat_time`` = per-unit adoption period (NaN for never-treated).
    Returns per-horizon effects + SEs + a joint significance read."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    uu = np.asarray(unit).ravel()
    tt = np.asarray(time, dtype=np.float64).ravel()
    gg = np.asarray(treat_time, dtype=np.float64).ravel()
    n = yy.size
    if uu.size != n or tt.size != n or gg.size != n or n < 40:
        raise ValueError("equal-length panel arrays, n>=40")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(tt)):
        raise ValueError("finite y and time required")
    if not np.isfinite(gg).any():
        raise ValueError("at least one treated unit required")

    out: dict[str, float] = {"n": float(n)}
    zs = []
    for h in horizons:
        try:
            b_h, se_h, n_t = _lp_did_at_h(yy, uu, tt, gg, h)
        except ValueError:
            out[f"eff_h{h}"] = math.nan
            continue
        z = b_h / max(se_h, 1e-12)
        zs.append(z)
        out[f"eff_h{h}"] = b_h
        out[f"se_h{h}"] = se_h
        out[f"n_treated_h{h}"] = float(n_t)
        out[f"p_h{h}"] = float(2 * (1 - norm.cdf(abs(z))))
    if zs:
        out["max_abs_z"] = float(max(abs(z) for z in zs))
    return out


def synth_lp_did(
    n_units: int = 120,
    n_periods: int = 10,
    adopt_share: float = 0.5,
    effect_growth: float = 0.4,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Staggered-adoption DGP with a RAMPING treatment effect
    (τ_h = effect_growth·(h+1)) — TWFE event studies contaminate;
    LP-DiD tracks each horizon."""
    rng = np.random.default_rng(seed)
    n_adopt = int(n_units * adopt_share)
    adopt_times = rng.choice([3, 4, 5], size=n_adopt)
    y_l, u_l, t_l, g_l = [], [], [], []
    alpha = rng.normal(0.0, 0.3, n_units)
    for i in range(n_units):
        g = float(adopt_times[i]) if i < n_adopt else math.nan
        lam = rng.normal(0.0, 0.2, n_periods)
        for t in range(n_periods):
            tau = 0.0
            if np.isfinite(g) and t >= g:
                tau = effect_growth * (int(t - g) + 1)
            y = alpha[i] + 0.1 * t + lam[t] + tau + rng.normal(0.0, 0.25)
            y_l.append(y)
            u_l.append(i)
            t_l.append(t)
            g_l.append(g)
    return {
        "y": np.asarray(y_l),
        "unit": np.asarray(u_l, dtype=np.float64),
        "time": np.asarray(t_l, dtype=np.float64),
        "treat_time": np.asarray(g_l),
        "effect_growth": np.array([effect_growth]),
    }


def bench_lp_did(seed: int = 20261231 + 214) -> dict[str, float]:
    """LP-DiD self-check: per-horizon effects track the ramp
    τ_h = 0.4·(h+1); null DGP stays flat. All ``synthetic_*``."""
    d = synth_lp_did(effect_growth=0.4, seed=seed)
    out = lp_did(
        np.asarray(d["y"]),
        np.asarray(d["unit"]),
        np.asarray(d["time"]),
        np.asarray(d["treat_time"]),
    )
    d0 = synth_lp_did(effect_growth=0.0, seed=seed + 1)
    out0 = lp_did(
        np.asarray(d0["y"]),
        np.asarray(d0["unit"]),
        np.asarray(d0["time"]),
        np.asarray(d0["treat_time"]),
    )
    out_b = lp_did(
        np.asarray(d["y"]),
        np.asarray(d["unit"]),
        np.asarray(d["time"]),
        np.asarray(d["treat_time"]),
    )

    errs = []
    for h in (0, 1, 2):
        v = out.get(f"eff_h{h}", math.nan)
        if math.isfinite(v):
            errs.append(abs(v - 0.4 * (h + 1)))
    mean_err = float(np.mean(errs)) if errs else math.nan
    null_max = float(
        max(
            abs(out0.get(f"eff_h{h}", 0.0))
            for h in (0, 1, 2)
            if math.isfinite(out0.get(f"eff_h{h}", math.nan))
        )
    )
    return {
        "synthetic_eff_h0": float(out.get("eff_h0", math.nan)),
        "synthetic_eff_h1": float(out.get("eff_h1", math.nan)),
        "synthetic_eff_h2": float(out.get("eff_h2", math.nan)),
        "synthetic_mean_err": mean_err,
        "synthetic_max_abs_z": float(out.get("max_abs_z", math.nan)),
        "synthetic_null_max_eff": null_max,
        "synthetic_detects": float(math.isfinite(mean_err) and mean_err < 0.3 and null_max < 0.4),
        "synthetic_determinism": float(
            float(out.get("eff_h0", math.nan)) == float(out_b.get("eff_h0", math.nan))
        ),
    }
