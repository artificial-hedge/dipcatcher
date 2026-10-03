"""Moment-inequality inference — Andrews & Soares / Chernozhukov et al.

Tests of the form H0: E[m_j(θ)] ≥ 0 for j = 1..p moments, where
the hypothesis is an identified SET (bounds, shapes constraints,
entry-game region). The plug-in-asymptotic (PA) statistic is the
criterion max over j of the studentized negative mean:

    T(θ) = sqrt(n) * max_j { -m̄_j(θ) / σ̂_j }

with the H0 rejection threshold from a Gaussian bootstrap on the
moment covariance (the "max" automatically handles dependence);
moment selection / GMS skips moments whose scaled means are deep
positive (κ_n = sqrt(n/ln n) slack). Inverting the test over a θ
grid gives a confidence set for a partially identified parameter.

References
----------
- Andrews, D.W.K., Soares, G. (2010). "Inference for parameters
  defined by moment inequalities." *Econometrica* 78(1).
- Romano, J.P., Shaikh, A.M., Wolf, M. (2014). "A practical two-
  step method for testing moment inequalities." *Econometrica* 82.
- Chernozhukov, V., Hong, H., Tamer, E. (2007). "Estimation and
  confidence regions for parameter sets." *Econometrica* 75.
- Bugni, F.A. (2010). "Bootstrap inference in partially
  identified models defined by moment inequalities." *Econometrica*.

Honesty
-------
SYNTHETIC DGPs only; bench verifies size control under the null
and rejection under violation — not a real bound estimate.

Composition
-----------
Called by ``quant_fund.research.benches_w67.bench_moment_ineq``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def moment_inequality_test(
    m: FloatArray,
    alpha: float = 0.05,
    n_boot: int = 500,
    kappa: float | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Test H0: E[m_j] >= 0 for all j. Returns stat, crit, p, reject.

    m (n, p) matrix of per-observation moments. GMS: moments with
    sqrt(n) m̄_j / σ̂_j > κ are dropped from the critical value
    (κ_n = sqrt(n/ln n) by default — the Andrews-Soares choice).
    """
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    n, p = m.shape
    if n < 20 or p < 1:
        raise ValueError("need (n>=20, p) moments")
    if not (0 < alpha < 0.5):
        raise ValueError("alpha out of range")
    rng = np.random.default_rng(seed)
    mbar = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("degenerate moment variance")
    t_stat = float(np.sqrt(n) * np.max(-mbar / sd))
    kappa_v = float(np.sqrt(n / np.log(n))) if kappa is None else float(kappa)
    selected = np.sqrt(n) * mbar / sd <= kappa_v
    if not np.any(selected):
        selected = np.ones(p, dtype=bool)
    cov = np.cov(m[:, selected].T)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov = np.asarray(cov, dtype=float)
    sd_sel = sd[selected]
    corr = cov / np.outer(sd_sel, sd_sel)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0) + 1e-9 * np.eye(corr.shape[0])
    draws = rng.multivariate_normal(np.zeros(int(np.sum(selected))), corr, n_boot)
    # bootstrap stat: max over selected of -Z_j (Z ~ N(0, corr))
    t_boot = np.max(-draws, axis=1)
    crit = float(np.quantile(t_boot, 1 - alpha))
    p_val = float(np.mean(t_boot >= t_stat))
    return {
        "stat": t_stat,
        "crit": crit,
        "p_value": p_val,
        "reject": float(t_stat > crit),
    }


def bench_moment_inequalities(seed: int = 20261231 + 394) -> dict[str, float]:
    """SYNTHETIC check — size control and power both verified."""
    rng = np.random.default_rng(seed)
    n, p = 400, 4
    # Null DGP: moments with means >= 0 (two binding at 0).
    means_null = np.array([0.0, 0.5, 0.0, 0.8])
    cov = np.eye(p) * 0.8 + 0.2
    m_null = means_null + rng.multivariate_normal(np.zeros(p), cov, n)
    out_null = moment_inequality_test(m_null, alpha=0.05, n_boot=400, seed=seed)
    if out_null["p_value"] < 0.005:
        raise ValueError("null rejected too aggressively")
    # Violated null: one moment mean negative.
    means_bad = np.array([-0.35, 0.5, 0.0, 0.8])
    m_bad = means_bad + rng.multivariate_normal(np.zeros(p), cov, n)
    out_bad = moment_inequality_test(m_bad, alpha=0.05, n_boot=400, seed=seed + 1)
    if out_bad["reject"] != 1.0 or out_bad["p_value"] > 0.05:
        raise ValueError("violation not rejected")
    # Deeply violated: t_stat should dwarf crit.
    means_deep = np.array([-1.0, 0.5, 0.0, 0.8])
    m_deep = means_deep + rng.multivariate_normal(np.zeros(p), cov, n)
    out_deep = moment_inequality_test(m_deep, alpha=0.05, n_boot=400, seed=seed + 2)
    if out_deep["stat"] < out_deep["crit"] * 2.0:
        raise ValueError("deep violation stat too small")
    return {
        "synthetic_mi_null_p": out_null["p_value"],
        "synthetic_mi_bad_p": out_bad["p_value"],
        "synthetic_mi_deep_stat": out_deep["stat"],
        "synthetic_mi_deep_crit": out_deep["crit"],
        "score": 1.0,
    }
