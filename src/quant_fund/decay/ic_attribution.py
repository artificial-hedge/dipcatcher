"""IC attribution across asset groups (sectors, buckets, regimes).

Answers "where does the signal's cross-sectional predictive content come
from": per-group IC series, cross-group dispersion of group IC means, and a
permutation confidence check for the best-minus-worst group spread.

- ``group_ic_series`` — per-period IC computed within each group of assets;
- ``group_ic_summary`` — mean/std/t-stat per group plus the spread between
  the best and worst group means;
- ``group_spread_permutation`` — permutation p-value for the best-minus-
  worst group IC spread against the null of exchangeable group labels.

Honesty: group structure is supplied by the caller; attribution is a
descriptive decomposition of the supplied panel.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — the
  fundamental law decomposed by universe segments.
- Politis, D. N., Romano, J. P. (1994). The stationary bootstrap.

Composition: numpy + scipy (locked); deterministic seeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.decay.ic_series import spearman_ic

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def group_ic_series(
    pred: FloatArray, actual: FloatArray, groups: IntArray
) -> dict[int, FloatArray]:
    """Per-period rank IC computed within each group.

    ``groups`` is an (N,) integer label per asset. Groups with fewer than
    3 assets in a period contribute NaN for that period.
    """
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if pred.ndim != 2 or actual.shape != pred.shape:
        raise ValueError("pred and actual must be (T, N) of equal shape")
    if groups.shape != (pred.shape[1],):
        raise ValueError("groups must be (N,)")
    out: dict[int, FloatArray] = {}
    for g in np.unique(groups):
        cols = np.flatnonzero(groups == g)
        if len(cols) < 3:
            continue
        ic = spearman_ic(pred[:, cols], actual[:, cols])
        out[int(g)] = ic
    return out


def group_ic_summary(series: dict[int, FloatArray]) -> dict[str, FloatArray | float]:
    """Per-group mean IC, t-stat, and the best-minus-worst spread."""
    if not series:
        raise ValueError("series must be non-empty")
    keys = sorted(series)
    means = np.array([float(np.nanmean(series[k])) for k in keys], dtype=np.float64)
    tstats = np.array(
        [
            float(
                np.nanmean(series[k])
                / (np.nanstd(series[k], ddof=1) + 1e-12)
                * np.sqrt(np.sum(np.isfinite(series[k])))
            )
            for k in keys
        ],
        dtype=np.float64,
    )
    i_best = int(np.argmax(means))
    i_worst = int(np.argmin(means))
    spread = float(means[i_best] - means[i_worst])
    return {
        "groups": np.asarray(keys, dtype=np.float64),
        "mean_ic": means,
        "tstat": tstats,
        "best_group": float(keys[i_best]),
        "worst_group": float(keys[i_worst]),
        "spread": spread,
    }


def group_spread_permutation(
    pred: FloatArray,
    actual: FloatArray,
    groups: IntArray,
    *,
    n_perm: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Permutation p-value for the best-minus-worst group IC spread.

    H0: the group labels carry no IC information. Each permutation re-draws
    the asset→group assignment at random and recomputes the spread; the
    p-value is the fraction of permuted spreads at least as large as the
    observed one. Valid when labels are exchangeable across assets.
    """
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if pred.ndim != 2 or actual.shape != pred.shape:
        raise ValueError("pred and actual must be (T, N) of equal shape")
    if groups.shape != (pred.shape[1],):
        raise ValueError("groups must be (N,)")
    if len(np.unique(groups)) < 2:
        raise ValueError("need at least two groups")
    if n_perm < 100:
        raise ValueError("n_perm must be >= 100 for a stable p-value")
    obs_series = group_ic_series(pred, actual, groups)
    obs = float(group_ic_summary(obs_series)["spread"])
    rng = np.random.default_rng(seed)
    count = 0
    n_assets = pred.shape[1]
    for _ in range(n_perm):
        perm = rng.permutation(n_assets)
        permuted = group_ic_series(pred, actual, groups[perm])
        if len(permuted) < 2:
            continue
        if float(group_ic_summary(permuted)["spread"]) >= obs:
            count += 1
    return {"spread": obs, "p": float((count + 1) / (n_perm + 1))}
