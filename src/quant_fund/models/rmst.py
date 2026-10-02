"""Restricted mean survival time (RMST) estimation + comparison.

RMST_tau = E[min(T, tau)] = int_0^tau S(t) dt, estimated by
trapezoid-integrating the Kaplan-Meier curve up to the
restriction time ``tau``. The Greenwood-style variance is

    var = sum_i  A_i^2 * d_i / (Y_i * (Y_i - d_i))

over event times t_i <= tau, where A_i is the remaining
area under the curve from t_i to tau and d_i / Y_i the
events / at-risk counts (Uno et al. 2004; Royston &
Parmar 2013). Group differences use a z-test on the
independent-sample difference; ratios a Wald test on the
log scale.

Honesty: `bench_rmst` checks the estimator against the
analytic RMST of an exponential DGP on SYNTHETIC data and
that a shifted-arm comparison has the right sign — a
recoverability diagnostic, not a survival-analysis claim
on real subjects.

References
----------
* Royston & Parmar (2011) "The use of restricted mean
  survival time to estimate the treatment effect in
  randomized clinical trials", Stat Med 30(19).
* Royston & Parmar (2013) "Restricted mean survival time:
  an alternative to the hazard ratio...", BMC Med Res
  Methodol 13:152.
* Uno et al. (2004) "Moving beyond the hazard ratio in
  quantifying the between-group difference in survival
  analysis", JCO 32(22).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _km_curve(
    t: FloatArray, d: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Kaplan-Meier times / survival / (events, at-risk)."""
    order = np.argsort(t)
    tt = np.asarray(t, dtype=np.float64)[order]
    dd = np.asarray(d, dtype=np.float64)[order]
    uniq = np.unique(tt)
    surv = np.empty_like(uniq)
    n_ev = np.empty_like(uniq)
    n_risk = np.empty_like(uniq)
    s = 1.0
    for i, ut in enumerate(uniq):
        at = tt == ut
        n_ev[i] = float(dd[at].sum())
        n_risk[i] = float(at.sum() + (tt > ut).sum())
        if n_risk[i] > 0:
            s *= 1.0 - n_ev[i] / n_risk[i]
        surv[i] = s
    return uniq, surv, n_ev, n_risk


def rmst(t: FloatArray, d: FloatArray, tau: float) -> dict[str, float]:
    """RMST point estimate + SE for one arm up to ``tau``."""
    tt = np.asarray(t, dtype=np.float64)
    dd = np.asarray(d, dtype=np.float64)
    if tt.size < 3 or tau <= 0 or np.any(tt < 0) or not np.isin(dd, (0.0, 1.0)).all():
        raise ValueError("bad survival inputs")
    ut, sv, n_ev, n_risk = _km_curve(tt, dd)

    def _area(x0: float) -> float:
        """Integral of the KM step curve from x0 to tau."""
        ks = np.concatenate([[x0], ut[(ut > x0) & (ut < tau)], [tau]])
        ix = np.searchsorted(ut, ks, side="right") - 1
        vs = np.where(ix >= 0, sv[np.clip(ix, 0, len(sv) - 1)], 1.0)
        return float(np.trapezoid(vs, ks)) if ks.size >= 2 else 0.0

    area = _area(0.0)
    # variance via per-event residual areas (Uno 2004)
    var = 0.0
    for i in range(len(ut)):
        if ut[i] >= tau or n_ev[i] <= 0 or n_risk[i] <= n_ev[i]:
            continue
        a_i = _area(float(ut[i]))
        var += a_i * a_i * n_ev[i] / (n_risk[i] * (n_risk[i] - n_ev[i]))
    return {"rmst": area, "se": float(math.sqrt(var)), "tau": float(tau)}


def rmst_compare(
    t1: FloatArray,
    d1: FloatArray,
    t2: FloatArray,
    d2: FloatArray,
    tau: float,
) -> dict[str, float]:
    """Two-arm RMST difference (arm1 - arm2) and ratio, z-tests."""
    a = rmst(t1, d1, tau)
    b = rmst(t2, d2, tau)
    diff = a["rmst"] - b["rmst"]
    se_diff = math.hypot(a["se"], b["se"])
    z = diff / max(se_diff, 1e-12)
    p_diff = float(2.0 * stats.norm.sf(abs(z)))
    # Wald on log-ratio
    se_l1 = a["se"] / max(a["rmst"], 1e-12)
    se_l2 = b["se"] / max(b["rmst"], 1e-12)
    z_r = math.log(max(a["rmst"], 1e-12) / max(b["rmst"], 1e-12)) / math.hypot(se_l1, se_l2)
    p_ratio = float(2.0 * stats.norm.sf(abs(z_r)))
    return {
        "rmst1": a["rmst"],
        "rmst2": b["rmst"],
        "diff": float(diff),
        "se_diff": float(se_diff),
        "p_diff": p_diff,
        "ratio": float(a["rmst"] / max(b["rmst"], 1e-12)),
        "p_ratio": p_ratio,
        "tau": float(tau),
    }


def bench_rmst(seed: int = 20261231 + 468) -> dict[str, float]:
    """SYNTHETIC check — RMST recovers analytic exponential mean."""
    rng = np.random.default_rng(seed)
    n, lam, tau = 800, 0.5, 2.0
    t = rng.exponential(1.0 / lam, size=n)
    d = np.ones(n)
    out = rmst(t, d, tau)
    true_rmst = (1.0 - math.exp(-lam * tau)) / lam
    if abs(out["rmst"] - true_rmst) / true_rmst > 0.08:
        raise ValueError(f"rmst off: {out['rmst']} vs {true_rmst}")
    # two-arm: arm2 hazard halved -> rmst2 > rmst1, significant
    t2 = rng.exponential(2.0 / lam, size=n)
    comp = rmst_compare(t, np.ones(n), t2, np.ones(n), tau)
    if not (comp["diff"] < 0 and comp["p_diff"] < 0.05):
        raise ValueError(f"compare off: {comp}")
    return {
        "synthetic_rmst_rel_err": float(abs(out["rmst"] - true_rmst) / true_rmst),
        "synthetic_diff_p": comp["p_diff"],
        "score": 1.0,
    }
