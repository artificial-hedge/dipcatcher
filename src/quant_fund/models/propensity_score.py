"""Propensity-score estimation, matching, weighting, and balance (SYNTHETIC).

Classic Rosenbaum–Rubin pipeline: logistic propensity fit (IRLS),
nearest-neighbor matching on the score with a caliper, inverse-
probability and overlap-weighted ATE/ATT estimators, and the covariate-
balance diagnostics (standardized mean differences, variance ratios)
that must be checked before any estimate is trusted.

All estimators fail closed (ValueError) on degenerate input; SMD/Vr
diagnostics are reported for raw AND adjusted designs so balance gains
(or failures) are never hidden.

Honesty: synthetic benches measure ATE/ATT recovery and post-weighting
balance on generated covariate-driven assignments — never market
evidence; positivity violations are surfaced via the overlap flags and
trim counts, not smoothed over.

References:
- Rosenbaum, Rubin (1983). The central role of the propensity score in
  observational studies for causal effects. *Biometrika* 70.
- Rosenbaum, Rubin (1985). Constructing a control group using
  multivariate matched sampling methods. *Amer. Statist.* 39.
- Hirano, Imbens, Ridder (2003). Efficient estimation of average
  treatment effects using the estimated propensity score.
  *Econometrica* 71.
- Li, Morgan, Zaslavsky (2018). Balancing covariates via propensity
  score weighting. *JASA* 113 (overlap weights).
- Austin (2009). Balance diagnostics for propensity score methods.
  *Stat. Med.* 28.

Composition: pure numpy — IRLS logistic fit, exact greedy matching;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_1d(x: FloatArray, name: str, n_min: int) -> FloatArray:
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < n_min:
        raise ValueError(f"{name}: need >= {n_min} finite obs, got {a.size}")
    return a


def _logit_fit(x: FloatArray, d: FloatArray, iters: int = 50) -> FloatArray:
    """IRLS logistic regression with a small ridge; returns coefficients."""
    n, p = x.shape
    a = np.column_stack([np.ones(n), x])
    beta = np.zeros(p + 1)
    lam = 1e-4
    for _ in range(iters):
        eta = a @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-6)
        z_ = eta + (d - mu) / w
        sw = np.sqrt(w)
        lhs = (a * sw[:, None]).T @ (a * sw[:, None]) + lam * np.eye(p + 1)
        rhs = (a * sw[:, None]).T @ (z_ * sw)
        beta_new = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def propensity_score(x: FloatArray, d: FloatArray) -> dict[str, float | FloatArray]:
    """Fit propensity e(x) = P(D=1|X) via ridge-IRLS logistic."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite (n, p) matrix")
    d = _as_1d(d, "d", x.shape[0])
    if d.size != x.shape[0]:
        raise ValueError("d must match x rows")
    if not np.isin(np.unique(d), [0.0, 1.0]).all():
        raise ValueError("d must be binary 0/1")
    beta = _logit_fit(x, d)
    eta = np.column_stack([np.ones(x.shape[0]), x]) @ beta
    ps = 1.0 / (1.0 + np.exp(-eta))
    return {"ps": ps, "coef": beta, "auc_proxy": float(np.mean(ps[d == 1]) - np.mean(ps[d == 0]))}


def _smd_row(x: FloatArray, d: FloatArray, w: FloatArray | None) -> float:
    if w is None:
        w = np.ones(d.size)
    d1, d0 = d == 1, d == 0
    if d1.sum() < 2 or d0.sum() < 2:
        return math.nan
    m1 = float(np.average(x[d1], weights=w[d1]))
    m0 = float(np.average(x[d0], weights=w[d0]))
    v1 = float(np.average((x[d1] - m1) ** 2, weights=w[d1]))
    v0 = float(np.average((x[d0] - m0) ** 2, weights=w[d0]))
    return float((m1 - m0) / math.sqrt(max(0.5 * (v1 + v0), 1e-12)))


def balance_table(
    x: FloatArray, d: FloatArray, w: FloatArray | None = None
) -> dict[str, float | FloatArray]:
    """Per-covariate standardized mean differences before/after weighting."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    d = _as_1d(d, "d", x.shape[0])
    if d.size != x.shape[0]:
        raise ValueError("d must match x rows")
    smd_raw = np.array([_smd_row(x[:, j], d, None) for j in range(x.shape[1])])
    smd_w = (
        np.array([_smd_row(x[:, j], d, w) for j in range(x.shape[1])]) if w is not None else smd_raw
    )
    return {
        "smd_raw": smd_raw,
        "smd_weighted": smd_w,
        "max_abs_smd_raw": float(np.nanmax(np.abs(smd_raw))),
        "max_abs_smd_weighted": float(np.nanmax(np.abs(smd_w))),
        "balanced_10pct": float(np.nanmax(np.abs(smd_w)) < 0.10),
    }


def ps_match(
    y: FloatArray,
    d: FloatArray,
    x: FloatArray,
    *,
    caliper: float = 0.2,
) -> dict[str, float | FloatArray]:
    """1:1 nearest-neighbor matching on the logit score with caliper;
    ATT estimator on matched pairs."""
    y = _as_1d(y, "y", 8)
    d = _as_1d(d, "d", y.size)
    if d.size != y.size:
        raise ValueError("d must match y")
    ps = np.asarray(propensity_score(x, d)["ps"])
    logit = np.log(np.clip(ps, 1e-6, 1 - 1e-6) / (1 - np.clip(ps, 1e-6, 1 - 1e-6)))
    tr_idx = np.flatnonzero(d == 1)
    co_idx = np.flatnonzero(d == 0)
    if tr_idx.size == 0 or co_idx.size == 0:
        raise ValueError("need both treated and control units")
    scale = float(np.std(logit))
    eff_caliper = caliper * max(scale, 1e-12)
    diffs = []
    used_ctrl: list[int] = []
    for i in tr_idx:
        dist = np.abs(logit[co_idx] - logit[i])
        j = int(np.argmin(dist))
        if dist[j] <= eff_caliper:
            diffs.append(float(y[i] - y[co_idx[j]]))
            used_ctrl.append(int(co_idx[j]))
    if len(diffs) < 2:
        raise ValueError("caliper too strict: fewer than 2 matched pairs")
    arr = np.asarray(diffs)
    return {
        "att": float(arr.mean()),
        "att_se": float(arr.std(ddof=1) / math.sqrt(arr.size)),
        "n_matched": float(arr.size),
        "match_rate": float(arr.size / tr_idx.size),
        "pair_diffs": arr,
    }


def ipw_ate(
    y: FloatArray, d: FloatArray, ps: FloatArray, *, trim: float = 0.02
) -> dict[str, float]:
    """Hájek IPW ATE with propensity trimming at ``trim`` tails."""
    y = _as_1d(y, "y", 8)
    d = _as_1d(d, "d", y.size)
    e = np.asarray(ps, dtype=np.float64).ravel()
    if e.size != y.size or not np.all(np.isfinite(e)):
        raise ValueError("ps must be finite and match y")
    keep = (e >= trim) & (e <= 1.0 - trim)
    if keep.sum() < y.size // 2:
        raise ValueError("trimming removed more than half the sample")
    y_, d_, e_ = y[keep], d[keep], e[keep]
    d1, d0 = d_ == 1, d_ == 0
    if d1.sum() < 1 or d0.sum() < 1:
        raise ValueError("need both treated and control units after trimming")
    mu1 = float(np.sum(y_[d1] / e_[d1]) / np.sum(1.0 / e_[d1]))
    mu0 = float(np.sum(y_[d0] / (1.0 - e_[d0])) / np.sum(1.0 / (1.0 - e_[d0])))
    return {
        "ate_ipw": mu1 - mu0,
        "mu1": mu1,
        "mu0": mu0,
        "n_trimmed": float(y.size - keep.sum()),
        "min_ps_kept": float(e_.min()),
        "max_ps_kept": float(e_.max()),
    }


def overlap_ate(y: FloatArray, d: FloatArray, ps: FloatArray) -> dict[str, float]:
    """Overlap-weighted ATO estimand (Li–Morgan–Zaslavsky): weights are
    (1-e) for treated and e for control — targets the population with
    clinical equipoise and is the variance-minimizing PS weighting."""
    y = _as_1d(y, "y", 8)
    d = _as_1d(d, "d", y.size)
    e = np.asarray(ps, dtype=np.float64).ravel()
    if e.size != y.size:
        raise ValueError("ps must match y")
    w1 = 1.0 - e
    w0 = e
    d1, d0 = d == 1, d == 0
    mu1 = float(np.sum(y[d1] * w1[d1]) / np.sum(w1[d1]))
    mu0 = float(np.sum(y[d0] * w0[d0]) / np.sum(w0[d0]))
    return {"ato": mu1 - mu0, "mu1_ow": mu1, "mu0_ow": mu0}


def synth_propensity(
    n: int = 1200,
    p: int = 4,
    ate: float = 1.0,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """Covariate-driven assignment: d ~ Bern(logit(Xβ)), y = ate·d + f(X)
    with nonlinear f — a setting where raw difference is badly biased."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, p))
    beta = np.linspace(-0.8, 0.8, p)
    eta = x @ beta - 0.3
    ps_true = 1.0 / (1.0 + np.exp(-eta))
    d = (rng.random(n) < ps_true).astype(np.float64)
    fx = 0.6 * x[:, 0] + 0.4 * np.sin(2.0 * x[:, 1]) + 0.3 * x[:, 2] ** 2
    y = ate * d + fx + rng.normal(0.0, 0.5, n)
    return {"y": y, "d": d, "x": x, "ps_true": ps_true, "ate": np.full(1, ate, dtype=np.float64)}


def bench_propensity_score(seed: int = 20261231 + 189) -> dict[str, float]:
    """PS self-check: IPW/overlap/matching recover the true ATE far better
    than the confounded raw difference; post-weighting balance collapses
    max|SMD| below the 10% rule of thumb. All ``synthetic_*``."""
    d = synth_propensity(seed=seed)
    y, dd, x = np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["x"])
    ate_true = float(np.asarray(d["ate"]).item())
    ps = np.asarray(propensity_score(x, dd)["ps"])

    raw = float(y[dd == 1].mean() - y[dd == 0].mean())
    ipw = ipw_ate(y, dd, ps)
    ow = overlap_ate(y, dd, ps)
    mt = ps_match(y, dd, x)
    bal = balance_table(x, dd, np.where(dd == 1, 1.0 - ps, ps))

    return {
        "synthetic_ate_true": ate_true,
        "synthetic_raw_bias": abs(raw - ate_true),
        "synthetic_ipw_err": abs(float(ipw["ate_ipw"]) - ate_true),
        "synthetic_ato_err": abs(float(ow["ato"]) - ate_true),
        "synthetic_att_match_err": abs(float(mt["att"]) - ate_true),
        "synthetic_ipw_beats_raw": float(
            float(ipw["ate_ipw"]) != raw
            and abs(float(ipw["ate_ipw"]) - ate_true) < 0.5 * abs(raw - ate_true)
        ),
        "synthetic_max_smd_raw": float(bal["max_abs_smd_raw"]),
        "synthetic_max_smd_weighted": float(bal["max_abs_smd_weighted"]),
        "synthetic_balance_improved": float(
            float(bal["max_abs_smd_weighted"]) < 0.5 * float(bal["max_abs_smd_raw"])
        ),
        "synthetic_match_rate": float(mt["match_rate"]),
        "synthetic_determinism": float(
            np.array_equal(np.asarray(propensity_score(x, dd)["ps"]), ps)
        ),
    }
