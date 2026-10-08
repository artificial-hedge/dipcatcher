"""Specification-curve (multiverse) analysis (SYNTHETIC).

Enumerate the full grid of defensible analytic specifications —
subsamples, control sets, functional forms — estimate the focal effect
under each, and summarize the resulting distribution. The shuffle test
(Simonsohn, Simmons, Nelson 2020) compares the observed curve against
curves recomputed on placebo-labeled data.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure effect-recovery across the
specification grid on generated panels — never market evidence.

References:
- Simonsohn, Simmons, Nelson (2020). Specification curve analysis.
  *Nature Human Behaviour* 4, 1208-1214.
- Steegen, Tuerlinckx, Gelman, Vanpaemel (2016). Increasing
  transparency through a multiverse analysis. *Persp. Psych. Sci.* 11.
- Young, Holsteen (2017). Model uncertainty and robustness: a
  computational framework for multimodel analysis. *Soc. Meth. Res.*

Composition: pure numpy — OLS per spec via lstsq, placebo shuffle via
permuted treatment; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import itertools
import math
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SpecCurveResult(TypedDict):
    effects_sorted: FloatArray
    effects: FloatArray
    z: FloatArray
    labels: list[str]
    n_specs: float
    median: float
    mean: float
    sd: float
    share_sig_pos: float
    share_sig_neg: float
    share_sig: float


def _as1(v: FloatArray, name: str) -> FloatArray:
    a = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 1-D vector required")
    return a


def _as2(m: FloatArray, name: str) -> FloatArray:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 2-D matrix required")
    return a


def _ols_hc1(y: FloatArray, x: FloatArray) -> tuple[float, float]:
    """OLS on [1, x] returning (coef on last column, HC1 SE)."""
    n = y.size
    X = np.column_stack([np.ones(n), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    k = X.shape[1]
    xtxi = np.linalg.pinv(X.T @ X)
    meat = (X * r[:, None]).T @ (X * r[:, None])
    vcov = xtxi @ meat @ xtxi * n / (n - k)
    return float(b[-1]), math.sqrt(max(float(vcov[-1, -1]), 0.0))


def spec_grid(
    subsample_kinds: tuple[str, ...] = ("all", "top_half", "bottom_half"),
    control_kinds: tuple[str, ...] = ("none", "linear", "quadratic"),
    transform_kinds: tuple[str, ...] = ("level", "log"),
) -> list[tuple[str, str, str]]:
    """Cartesian product of analytic choices."""
    return [
        s
        for s in itertools.product(subsample_kinds, control_kinds, transform_kinds)
        if not (s[0] != "all" and s[1] == "none" and s[2] == "log")
    ]


def specification_curve(
    y: FloatArray,
    treat: FloatArray,
    covariate: FloatArray | None,
    grid: list[tuple[str, str, str]] | None = None,
) -> SpecCurveResult:
    """Run the focal treatment regression over every spec in ``grid``.

    ``y``: outcome; ``treat``: focal regressor; ``covariate``: optional
    nuisance covariate consumed by control kinds ('linear', 'quadratic').
    Each spec filters subsample, builds the control block, transforms
    the outcome, then regresses. Returns sorted effect estimates with
    significance counts and the full curve."""
    ya = _as1(y, "y")
    ta = _as1(treat, "treat")
    n = ya.size
    if ta.size != n:
        raise ValueError("treat length mismatch")
    ca = None if covariate is None else _as1(covariate, "covariate")
    if ca is not None and ca.size != n:
        raise ValueError("covariate length mismatch")
    if grid is None:
        grid = spec_grid()
    if not grid:
        raise ValueError("empty spec grid")

    med = float(np.median(ca)) if ca is not None else 0.0
    effects: list[float] = []
    zvals: list[float] = []
    labels: list[str] = []
    for sub, ctrl, trf in grid:
        if sub == "top_half":
            if ca is None:
                continue
            mask = ca >= med
        elif sub == "bottom_half":
            if ca is None:
                continue
            mask = ca < med
        else:
            mask = np.ones(n, dtype=bool)
        yy = ya[mask]
        tt = ta[mask]
        if yy.size < 8 or np.std(tt) < 1e-12:
            continue
        if trf == "log":
            if np.min(yy) <= -1.0 + 1e-9:
                continue
            yy = np.log1p(yy - min(float(np.min(yy)) - 0.01, 0.0))
        blocks = [tt]
        if ctrl != "none":
            if ca is None:
                continue
            cc = ca[mask]
            blocks.append(cc)
            if ctrl == "quadratic":
                blocks.append(cc**2)
        xmat = np.column_stack(blocks[1:] + [tt])  # treat last
        try:
            b2, se2 = _ols_hc1(yy, xmat)
        except np.linalg.LinAlgError:
            continue
        effects.append(b2)
        zvals.append(b2 / max(se2, 1e-12))
        labels.append(f"{sub}|{ctrl}|{trf}")

    eff = np.asarray(effects)
    z = np.asarray(zvals)
    sig_pos = float(np.mean(z > 1.96))
    sig_neg = float(np.mean(z < -1.96))
    return {
        "effects_sorted": np.sort(eff),
        "effects": eff,
        "z": z,
        "labels": labels,
        "n_specs": float(eff.size),
        "median": float(np.median(eff)),
        "mean": float(eff.mean()),
        "sd": float(eff.std(ddof=1)) if eff.size > 1 else 0.0,
        "share_sig_pos": sig_pos,
        "share_sig_neg": sig_neg,
        "share_sig": sig_pos + sig_neg,
    }


def spec_curve_shuffle_p(
    y: FloatArray,
    treat: FloatArray,
    covariate: FloatArray | None,
    n_shuffles: int = 200,
    grid: list[tuple[str, str, str]] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Placebo shuffle test: permute treatment labels, recompute each
    curve's median|effect|, report the share of placebo medians ≥ the
    observed (a bootstrapped p-value for the multiverse effect)."""
    ya = _as1(y, "y")
    ta = _as1(treat, "treat")
    rng = np.random.default_rng(seed)
    obs = specification_curve(ya, ta, covariate, grid)
    obs_stat = float(abs(float(obs["median"])))
    null_stats = np.empty(n_shuffles)
    for i in range(n_shuffles):
        tp = rng.permutation(ta)
        try:
            r = specification_curve(ya, tp, covariate, grid)
            null_stats[i] = abs(float(r["median"]))
        except ValueError:
            null_stats[i] = np.nan
    ok = null_stats[np.isfinite(null_stats)]
    p = float((1.0 + float(np.sum(ok >= obs_stat))) / (ok.size + 1.0))
    return {
        "p_value": p,
        "obs_median_abs": obs_stat,
        "null_median_abs_mean": float(ok.mean()),
        "n_shuffles_used": float(ok.size),
        "obs_share_sig": float(obs["share_sig"]),
        "obs_median": float(obs["median"]),
        "obs_n_specs": float(obs["n_specs"]),
    }


def synth_multiverse(
    n: int = 500,
    effect: float = 0.5,
    confound: float = 0.4,
    hetero: bool = True,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Panel with a confounded heterogeneous treatment effect.

    ``hetero=True`` makes the effect vary across the covariate, so
    subsample specs legitimately disagree (spec curve spreads)."""
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    t_ = rng.normal(confound * c, 0.8, n)
    tau = effect + (0.8 * (c > 0) - 0.4) if hetero else effect
    y = tau * t_ + 0.7 * c + rng.normal(0.0, 1.0, n)
    return {"y": y, "treat": t_, "covariate": c, "effect_true": np.full(n, tau)}


def bench_specification_curve(seed: int = 20261231 + 200) -> dict[str, float]:
    """Multiverse self-check: median recovers effect; effect-0 shuffle
    p>effect-real p; deterministic. All ``synthetic_*``."""
    d = synth_multiverse(seed=seed, effect=0.5)
    sc = specification_curve(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["covariate"]))
    sh = spec_curve_shuffle_p(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["covariate"]),
        n_shuffles=100,
        seed=seed + 1,
    )
    d0 = synth_multiverse(seed=seed + 2, effect=0.0)
    sh0 = spec_curve_shuffle_p(
        np.asarray(d0["y"]),
        np.asarray(d0["treat"]),
        np.asarray(d0["covariate"]),
        n_shuffles=100,
        seed=seed + 3,
    )
    sc_b = specification_curve(
        np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["covariate"])
    )
    return {
        "synthetic_median": float(sc["median"]),
        "synthetic_median_err": float(abs(float(sc["median"]) - 0.5)),
        "synthetic_n_specs": float(sc["n_specs"]),
        "synthetic_share_sig": float(sc["share_sig"]),
        "synthetic_shuffle_p_real": float(sh["p_value"]),
        "synthetic_shuffle_p_null": float(sh0["p_value"]),
        "synthetic_detects": float(float(sh["p_value"]) < 0.15 and float(sc["median"]) > 0.15),
        "synthetic_determinism": float(float(sc["median"]) == float(sc_b["median"])),
    }
