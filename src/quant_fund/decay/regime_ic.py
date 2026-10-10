"""IC conditioning on an exogenous regime indicator.

Splits the IC series by a categorical regime (e.g. volatility state) and
quantifies how the signal's predictive content co-moves with the regime:

- ``regime_ic_summary`` — per-regime IC mean/std/t-stat and counts;
- ``regime_ic_difference`` — mean difference between two named regimes with
  a permutation two-sided p-value (labels re-assigned over time, preserving
  label frequencies);
- ``regime_ic_dispersion`` — cross-regime dispersion of regime IC means,
  normalised by the overall IC std.

Honesty: regimes are supplied by the caller; conditioning is descriptive.

References:
- Ang, A., Timmermann, A. (2012). Regime changes and financial markets —
  conditioning diagnostics on states.
- Politis, D. N., Romano, J. P. (1994). The stationary bootstrap.

Composition: numpy + scipy (locked); deterministic seeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _valid_pair(ic: FloatArray, regime: IntArray) -> tuple[FloatArray, IntArray]:
    x = np.asarray(ic, dtype=np.float64)
    r = np.asarray(regime, dtype=np.int64)
    if x.ndim != 1 or r.shape != x.shape:
        raise ValueError("ic and regime must be one-dimensional arrays of equal shape")
    mask = np.isfinite(x)
    return x[mask], r[mask]


def regime_ic_summary(ic: FloatArray, regime: IntArray) -> dict[str, dict[str, float]]:
    """Per-regime IC mean, std, t-stat and observation count."""
    x, r = _valid_pair(ic, regime)
    if len(x) < 6:
        raise ValueError("need at least 6 valid observations")
    out: dict[str, dict[str, float]] = {}
    for g in np.unique(r):
        xg = x[r == g]
        sd = float(np.std(xg, ddof=1)) if len(xg) > 1 else float("nan")
        t = float(np.mean(xg) / sd * np.sqrt(len(xg))) if sd and sd > 0 else float("nan")
        out[str(int(g))] = {
            "n": float(len(xg)),
            "mean": float(np.mean(xg)),
            "std": sd,
            "tstat": t,
        }
    return out


def regime_ic_difference(
    ic: FloatArray,
    regime: IntArray,
    regime_a: int,
    regime_b: int,
    *,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean IC in regime A minus regime B with a permutation p-value.

    H0: the regime label carries no IC information. Each permutation
    re-assigns the observed regime labels across time (breaking the
    regime↔IC alignment while preserving the label frequencies); the
    p-value is the two-sided fraction of permuted differences at least as
    extreme as the observed one. Valid when labels are exchangeable over
    time; for serially correlated regimes use a block permutation variant.
    """
    x, r = _valid_pair(ic, regime)
    xa = x[r == regime_a]
    xb = x[r == regime_b]
    if len(xa) < 5 or len(xb) < 5:
        raise ValueError("need at least 5 observations per regime")
    if n_perm < 100:
        raise ValueError("n_perm must be >= 100 for a stable p-value")
    obs = float(np.mean(xa) - np.mean(xb))
    n = len(x)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rp = r[rng.permutation(n)]
        if np.sum(rp == regime_a) < 3 or np.sum(rp == regime_b) < 3:
            continue
        diff_p = float(np.mean(x[rp == regime_a]) - np.mean(x[rp == regime_b]))
        if abs(diff_p) >= abs(obs):
            count += 1
    return {"diff": obs, "p": float((count + 1) / (n_perm + 1))}


def regime_ic_dispersion(ic: FloatArray, regime: IntArray) -> float:
    """Std of per-regime IC means / overall IC std (≥ 0)."""
    x, r = _valid_pair(ic, regime)
    if len(np.unique(r)) < 2:
        raise ValueError("need at least two regimes")
    means = np.array([np.mean(x[r == g]) for g in np.unique(r)], dtype=np.float64)
    overall = float(np.std(x, ddof=1))
    if overall <= 0:
        return 0.0
    return float(np.std(means, ddof=1) / overall)
