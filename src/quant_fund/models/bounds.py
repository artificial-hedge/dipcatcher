"""Partial-identification bounds under missing data and selection.

When point identification fails, the sharp answer is a set. Two
classical constructions:

1. **Manski bounds** (Manski 1990, 2003): with outcome missing at rate
   ``1 − π``, E[Y] is only bounded — replacing missing outcomes with
   the support endpoints ``[y_lo, y_hi]`` gives the sharp worst-case
   bound; adding a monotone-instrument or missing-at-random
   assumption tightens it.

2. **Lee bounds** (Lee 2009): under sample selection, the treatment
   effect on the "always-observed" subpopulation is bounded by
   trimming the excess selection share off the outcome tails of the
   more-selected arm.

Both are computed on finite samples with a simple asymptotic normal
interval for the bound endpoints (Horowitz–Manski / Lee).

References
----------
- Manski, C. F. (1990). *Nonparametric bounds on treatment effects.*
  American Economic Review 80(2), 319–323.
- Manski, C. F. (2003). *Partial Identification of Probability
  Distributions.* Springer.
- Lee, D. S. (2009). *Training, wages, and sample selection:
  estimating sharp bounds on treatment effects.* Review of Economic
  Studies 76(3), 1071–1102.
- Horowitz, J. L. & Manski, C. F. (2000). *Nonparametric analysis of
  randomized experiments with missing covariate and outcome data.*
  JASA 95(449), 77–84.

Honesty contract
----------------
``synth_*`` helpers and ``bench_*`` emit SYNTHETIC correctness checks
only — never market evidence; bound widths are reported honestly
(they can be wide; that is the point of the estimator).

Composition notes
-----------------
numpy/scipy only; deterministic ``np.random.default_rng(seed)``;
fail-closed ``ValueError`` on degenerate support or zero selection
share; public dict keys never contain forbidden score tokens.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

FloatArray = np.ndarray

__all__ = [
    "bench_bounds",
    "lee_bounds",
    "manski_bounds",
    "manski_mar_bounds",
    "synth_missing",
    "synth_selection",
]

_RT = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _z(phi: float) -> float:
    return float(stats.norm.ppf(1.0 - (1.0 - phi) / 2.0))


def manski_bounds(
    y_obs: FloatArray,
    missing: FloatArray,
    y_lo: float | None = None,
    y_hi: float | None = None,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Worst-case bound on E[Y] with outcomes missing at rate 1−π.

    ``missing`` is a 0/1 (or bool) array aligned with ``y_obs`` marking
    which units lack outcome data. ``y_lo``/``y_hi`` bound the outcome
    support; when omitted they are read off the observed range — a
    sample plug-in, flagged via ``support_estimated``.

    Returns the Manski interval [LB, UB], its width, the missingness
    share, and a Horowitz–Manski normal interval.
    """
    y = np.asarray(y_obs, dtype=np.float64)
    m = np.asarray(missing, dtype=np.float64)
    if y.ndim != 1 or m.ndim != 1 or m.shape[0] != y.shape[0]:
        raise ValueError("y_obs and missing must be same-length vectors")
    n_obs = y.size
    if n_obs < 20:
        raise ValueError("need at least 20 observations")
    n_miss = int(np.count_nonzero(m > 0.5))
    n_seen = n_obs - n_miss
    if n_seen < 10:
        raise ValueError("fewer than 10 observed outcomes")
    p_obs = n_seen / n_obs
    if y_lo is None or y_hi is None:
        support_est = 1.0
        seen = y[m <= 0.5]
        lo = float(seen.min())
        hi = float(seen.max())
    else:
        support_est = 0.0
        lo = float(y_lo)
        hi = float(y_hi)
    if not lo < hi:
        raise ValueError("outcome support is degenerate")
    mu = float(y[m <= 0.5].mean())
    lb = p_obs * mu + (1.0 - p_obs) * lo
    ub = p_obs * mu + (1.0 - p_obs) * hi
    # Horowitz–Manski: each endpoint's se from a binomial-scale view;
    # conservative se ≈ (hi−lo)·sqrt(Var(p̂)+Var(ȳ|obs)/n_seen).
    se_lb = (hi - lo) * math.sqrt(
        p_obs * (1 - p_obs) / n_obs + float(y[m <= 0.5].var()) / max(n_seen, 1) / 2
    )
    se_ub = se_lb
    z = _z(alpha)
    return {
        "lb": lb,
        "ub": ub,
        "width": ub - lb,
        "p_observed": p_obs,
        "mean_observed": mu,
        "se_lb": se_lb,
        "se_ub": se_ub,
        "ci_lb": lb - z * se_lb,
        "ci_ub": ub + z * se_ub,
        "support_estimated": support_est,
    }


def manski_mar_bounds(
    y_obs: FloatArray,
    missing: FloatArray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bound under missing-at-random: [E[Y|obs], E[Y|obs]] collapses to
    a point (MAR identifies the mean) — reported for contrast alongside
    the unconstrained Manski width. Also returns the observed-mean and
    its CI.
    """
    y = np.asarray(y_obs, dtype=np.float64)
    m = np.asarray(missing, dtype=np.float64)
    if y.shape != m.shape or y.ndim != 1:
        raise ValueError("y_obs and missing must be same-length vectors")
    seen = y[m <= 0.5]
    if seen.size < 10:
        raise ValueError("fewer than 10 observed outcomes")
    mu = float(seen.mean())
    se = float(seen.std(ddof=1)) / math.sqrt(seen.size)
    z = _z(alpha)
    return {
        "mar_mean": mu,
        "mar_se": se,
        "mar_ci_lb": mu - z * se,
        "mar_ci_ub": mu + z * se,
        "n_observed": float(seen.size),
    }


def lee_bounds(
    y: FloatArray,
    treat: FloatArray,
    observed: FloatArray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Lee (2009) sharp bounds on the treatment effect for
    always-observed compliers.

    Requires the treated arm to select *more* (standard orientation;
    swap labels if the control arm selects more). Excess selection
    share ``p = (p1 − p0)/p1`` is trimmed off the outcome tails of the
    more-selected arm: the lower bound trims from the top (removes the
    best outcomes → lowest plausible mean) and the upper bound trims
    from the bottom.

    Returns bound endpoints, the excess-selection share, per-arm
    selection rates, and a normal interval on the midpoint.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(treat, dtype=np.float64)
    s = np.asarray(observed, dtype=np.float64)
    if y.ndim != 1 or t.shape != y.shape or s.shape != y.shape:
        raise ValueError("y, treat, observed must be same-length vectors")
    if y.size < 60:
        raise ValueError("need at least 60 observations")
    y1 = y[(t > 0.5) & (s > 0.5)]
    y0 = y[(t <= 0.5) & (s > 0.5)]
    p1 = float((s[t > 0.5]).mean())
    p0 = float((s[t <= 0.5]).mean())
    if p1 <= p0 + 1e-9:
        raise ValueError("treated arm must select strictly more than control")
    if y1.size < 20 or y0.size < 20:
        raise ValueError("too few selected units in an arm")
    excess = (p1 - p0) / p1
    q = float(np.quantile(y1, excess))
    y1_lb = y1[y1 <= q]
    y1_ub = y1[y1 >= q]
    if y1_lb.size < 5 or y1_ub.size < 5:
        raise ValueError("trimmed arm too small")
    lb = float(y1_lb.mean() - y0.mean())
    ub = float(y1_ub.mean() - y0.mean())
    mid = 0.5 * (lb + ub)
    se_mid = math.sqrt(float(y1.var(ddof=1)) / y1.size + float(y0.var(ddof=1)) / y0.size)
    z = _z(alpha)
    return {
        "lb": lb,
        "ub": ub,
        "mid": mid,
        "width": ub - lb,
        "se_mid": se_mid,
        "ci_lb": mid - z * se_mid,
        "ci_ub": mid + z * se_mid,
        "excess_share": excess,
        "p_sel_treat": p1,
        "p_sel_ctrl": p0,
        "trim_quantile": q,
    }


def synth_missing(
    n: int = 1200,
    mu: float = 2.0,
    miss_frac: float = 0.3,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC Y ~ N(mu, 1) with MCAR missingness; returns observed
    outcome (nan where missing), missingness flag, and true mean."""
    rng = np.random.default_rng(seed)
    y = mu + rng.standard_normal(n)
    m = (rng.random(n) < miss_frac).astype(np.float64)
    y_obs = np.where(m > 0.5, np.nan, y)
    return {"y": y, "y_obs": y_obs, "missing": m, "mu_true": np.float64(mu)}


def synth_selection(
    n: int = 1500,
    tau: float = 1.0,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC Lee design: control selects 50%, treated selects 80%;
    outcome = tau·treat + sel-correlated noise + base."""
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < 0.5).astype(np.float64)
    u = rng.standard_normal(n)
    s_prob = 0.5 + 0.3 * t + 0.1 * u
    s = (rng.random(n) < np.clip(s_prob, 0, 1)).astype(np.float64)
    y = 1.0 + tau * t + 0.8 * u + 0.3 * rng.standard_normal(n)
    return {"y": y, "treat": t, "observed": s, "tau_true": np.float64(tau)}


def bench_bounds(seed: int = 20261231 + 178) -> dict[str, float]:
    """SYNTHETIC: Manski width covers truth, Lee bounds cover tau."""
    dm = synth_missing(n=1200, mu=2.0, miss_frac=0.3, seed=seed)
    y_obs = np.asarray(dm["y_obs"])
    seen = ~np.isnan(y_obs)
    mb = manski_bounds(
        np.where(seen, y_obs, 0.0), 1.0 - seen.astype(np.float64), y_lo=-3.0, y_hi=7.0
    )
    mar = manski_mar_bounds(np.where(seen, y_obs, 0.0), 1.0 - seen.astype(np.float64))
    ds = synth_selection(n=1500, tau=1.0, seed=seed + 1)
    lb = lee_bounds(np.asarray(ds["y"]), np.asarray(ds["treat"]), np.asarray(ds["observed"]))
    covers_m = float(mb["lb"] <= 2.0 <= mb["ub"])
    covers_l = float(lb["lb"] <= 1.0 <= lb["ub"])
    mb2 = manski_bounds(
        np.where(seen, y_obs, 0.0), 1.0 - seen.astype(np.float64), y_lo=-3.0, y_hi=7.0
    )
    det = float(mb["lb"] == mb2["lb"] and mb["ub"] == mb2["ub"])
    return {
        "synthetic_manski_width": float(mb["width"]),
        "synthetic_manski_covers": covers_m,
        "synthetic_mar_err": abs(float(mar["mar_mean"]) - 2.0),
        "synthetic_lee_width": float(lb["width"]),
        "synthetic_lee_lb": float(lb["lb"]),
        "synthetic_lee_ub": float(lb["ub"]),
        "synthetic_lee_covers": covers_l,
        "synthetic_lee_excess": float(lb["excess_share"]),
        "synthetic_determinism": det,
    }
