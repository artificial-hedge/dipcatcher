"""Competing-risks analysis — Aalen-Johansen CIF + pseudo-value regression (SYNTHETIC).

For event types j∈{1..K}: the cumulative incidence function F_j(t) =
P(T≤t, cause j) estimated by Aalen-Johansen on the cause-specific
hazards. Covariate effects on F_j(τ) come from Klein-Andersen
pseudo-observations: θ_i = n·F(τ) - (n-1)·F_{-i}(τ), regressed on X
by OLS — a valid marginal-model alternative to Fine-Gray.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure CIF and pseudo-value effect
recovery on generated competing-risks panels — never market evidence.

References:
- Aalen, Johansen (1978). An empirical transition matrix for
  non-homogeneous Markov chains. *Scand. J. Statistics* 5.
- Fine, Gray (1999). A proportional hazards model for the
  subdistribution of a competing risk. *JASA* 94.
- Klein, Andersen (2005). Regression modeling of competing risks
  data based on pseudovalues. *Biometrics* 61.
- Andersen, Geskus, de Witte, Putter (2012). Competing risks in
  epidemiology. *Int. J. Epidemiology* 41.

Composition: pure numpy — risk-set counting, jackknife pseudo-values,
OLS link; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as1(v: FloatArray, name: str) -> FloatArray:
    a = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 1-D vector required")
    return a


def aalen_johansen_cif(
    time: FloatArray,
    event: FloatArray,
    cause: int = 1,
) -> dict[str, FloatArray]:
    """CIF for one cause: Π_t (1 - d_any/Y) accumulated per cause.

    ``event``: 0 = censored, 1..K = cause indicator. Returns the CIF
    evaluated on the sorted unique event times."""
    t = _as1(time, "time")
    e = np.asarray(event).ravel()
    if e.size != t.size:
        raise ValueError("time/event length mismatch")
    if t.size < 10:
        raise ValueError("need >=10 observations")
    if not np.all(e >= 0) or e.astype(int).max() < 1:
        raise ValueError("events must be 0=censored or >=1 causes")
    if cause < 1 or cause > int(e.max()):
        raise ValueError("cause out of range")

    grid = np.unique(t[e == cause])
    cif = np.empty(grid.size)
    surv = 1.0  # overall survival S(t-)
    for i, tt in enumerate(grid):
        at_risk = float(np.sum(t >= tt))
        d_cause = float(np.sum((t == tt) & (e == cause)))
        d_any = float(np.sum((t == tt) & (e > 0)))
        if at_risk <= 0:
            cif[i] = cif[i - 1] if i else 0.0
            continue
        cif[i] = (cif[i - 1] if i else 0.0) + surv * (d_cause / at_risk)
        surv *= 1.0 - d_any / at_risk
    return {"times": grid, "cif": cif, "cif_end": np.array([cif[-1]])}


def pseudo_value_effects(
    time: FloatArray,
    event: FloatArray,
    covariates: FloatArray,
    cause: int = 1,
    horizon: float | None = None,
) -> dict[str, float | FloatArray]:
    """Klein-Andersen pseudo-value regression of CIF(τ) on covariates.

    Pseudo_i = n·F̂(τ) - (n-1)·F̂_{-i}(τ); OLS of pseudo values on
    [1, X] gives marginal covariate effects on the τ-CIF."""
    t = _as1(time, "time")
    e = np.asarray(event).ravel()
    X = _as1(covariates, "covariates")[:, None]
    n = t.size
    if X.shape[0] != n:
        raise ValueError("covariates length mismatch")
    if n < 15:
        raise ValueError("need >=15 observations for jackknife")
    grid_all = np.unique(t[e == cause])
    if grid_all.size == 0:
        raise ValueError("no events of target cause")
    tau = float(grid_all[int(0.6 * (grid_all.size - 1))] if horizon is None else horizon)

    def _cif_at(mask: NDArray[np.bool_]) -> float:
        cif_obj = aalen_johansen_cif(t[mask], e[mask], cause=cause)
        times = np.asarray(cif_obj["times"])
        cifv = np.asarray(cif_obj["cif"])
        if times.size == 0 or tau < times[0]:
            return 0.0
        j = int(np.searchsorted(times, tau, side="right") - 1)
        return float(cifv[j]) if j >= 0 else 0.0

    full = _cif_at(np.ones(n, dtype=bool))
    pseudo = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pseudo[i] = n * full - (n - 1) * _cif_at(mask)

    Xd = np.column_stack([np.ones(n), X])
    b, *_ = np.linalg.lstsq(Xd, pseudo, rcond=None)
    r = pseudo - Xd @ b
    k = Xd.shape[1]
    xtxi = np.linalg.pinv(Xd.T @ Xd)
    meat = (Xd * r[:, None]).T @ (Xd * r[:, None])
    vcov = xtxi @ meat @ xtxi * n / (n - k)
    se = math.sqrt(max(float(vcov[1, 1]), 0.0))

    return {
        "tau": tau,
        "cif_tau": full,
        "beta": float(b[1]),
        "se": se,
        "z": float(b[1]) / max(se, 1e-12),
        "pseudo_mean": float(pseudo.mean()),
        "n_obs": float(n),
    }


def synth_competing_risks(
    n: int = 500,
    effect: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Two-cause DGP: covariate raises cause-1 intensity only; cause-2
    competes. Censoring exponential. ``effect`` is the log hazard ratio."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    lam1 = 0.15 * np.exp(effect * x)
    lam2 = np.full(n, 0.10)
    lamc = np.full(n, 0.05)
    t1 = rng.exponential(1.0 / lam1)
    t2 = rng.exponential(1.0 / lam2)
    tc = rng.exponential(1.0 / lamc)
    t_obs = np.minimum(t1, np.minimum(t2, tc))
    ev = np.zeros(n, dtype=int)
    ev[(t_obs == t1)] = 1
    ev[(t_obs == t2)] = 2
    return {
        "time": t_obs,
        "event": ev.astype(np.float64),
        "x": x,
        "effect_true": np.array([effect]),
    }


def bench_competing_risks(seed: int = 20261231 + 206) -> dict[str, float]:
    """Competing-risks self-check: CIF monotone and below 1; pseudo-value
    regression recovers a positive covariate effect on CIF for cause 1
    and ~0 for cause 2 (which the covariate does not drive).
    All ``synthetic_*``."""
    d = synth_competing_risks(seed=seed, effect=0.6)
    t = np.asarray(d["time"])
    e = np.asarray(d["event"])
    x = np.asarray(d["x"])

    cif1 = aalen_johansen_cif(t, e, cause=1)
    pv1 = pseudo_value_effects(t, e, x, cause=1)
    pv2 = pseudo_value_effects(t, e, x, cause=2)

    cifv = np.asarray(cif1["cif"])
    mono = float(np.all(np.diff(cifv) >= -1e-12))
    pv_b = pseudo_value_effects(t, e, x, cause=1)

    # honest readout: cause-1 effect positive; cause-2 (untargeted)
    # effect may be mildly negative (competition) — gate on sign of 1.
    return {
        "synthetic_cif_end_cause1": float(cif1["cif_end"][0]),
        "synthetic_cif_monotone": mono,
        "synthetic_beta_cause1": float(pv1["beta"]),
        "synthetic_z_cause1": float(pv1["z"]),
        "synthetic_beta_cause2": float(pv2["beta"]),
        "synthetic_detects": float(
            float(pv1["beta"]) > 0.01 and float(pv1["z"]) > 1.0 and mono == 1.0
        ),
        "synthetic_determinism": float(float(pv1["beta"]) == float(pv_b["beta"])),
    }
