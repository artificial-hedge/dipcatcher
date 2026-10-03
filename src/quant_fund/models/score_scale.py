"""Two-sample rank tests for SCALE differences.

Canonical references:

- Ansari & Bradley (1960) 'Rank-sum tests for
  dispersions' AMS 31 — folded ranks about the pooled
  median; the location-shift-free baseline.
- Mood (1954) 'On the asymptotic efficiency of certain
  nonparametric two-sample tests' AMS 25 — sum of
  squared centered ranks.
- Klotz (1962) 'Nonparametric tests for scale' AMS 33 —
  normal-scores squared scale test, asymptotically
  optimal vs normal alternatives.
- Savage (1956) 'Contributions to the theory of rank
  order statistics' AMS 27 — exponential-scores test
  optimal vs exponential/leptokurtic alternatives.
- Capon (1961) 'Asymptotic efficiency of certain
  locally most powerful rank tests' AMS 32 — squared
  normal scores (Klotz-equivalent power).
- Gastwirth (1965) 'Percentile modifications of two
  sample rank tests' JASA 60 — winsorized-score
  compromise robust to outliers.

Statistics are the group-1 score sums; z via the exact
null mean/variance (no ties correction).

`bench_score_scale`: N(0,1) vs N(0,2) power ~1 on all;
equal-scale accepted ~95%.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _check2(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if xa.size < 8 or ya.size < 8:
        raise ValueError("samples <8")
    if not np.isfinite(xa).all() or not np.isfinite(ya).all():
        raise ValueError("non-finite")
    return xa, ya


def _rank_test(x: FloatArray, y: FloatArray, scores: FloatArray) -> dict[str, float]:
    """Sum of scores in group x; normal approx with the
    exact null mean/variance of a simple random sample."""
    xa, ya = _check2(x, y)
    n, m = xa.size, ya.size
    n_tot = n + m
    pooled = np.concatenate([xa, ya])
    ranks = stats.rankdata(pooled)
    a = np.asarray(scores, dtype=np.float64)
    if a.size != n_tot:
        raise ValueError("scores wrong size")
    # scores as a function of rank: a[i] applies to rank i
    ax = a[ranks[:n].astype(int) - 1]
    stat = float(ax.sum())
    e = n * float(a.mean())
    var = n * m / (n_tot * (n_tot - 1)) * float(((a - a.mean()) ** 2).sum())
    if var <= 0:
        raise ValueError("degenerate scores")
    z = (stat - e) / np.sqrt(var)
    return {
        "stat": stat,
        "z": float(z),
        "pvalue": float(2 * stats.norm.sf(abs(z))),
    }


def ansari_bradley(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Ansari-Bradley (1960) folded-rank scale test.

    Pool + rank the raw samples; score each rank r as
    (N+1)/2 - |r - (N+1)/2| so extreme ranks get small
    scores; statistic = group-1 score sum."""
    xa, ya = _check2(x, y)
    n_tot = xa.size + ya.size
    r = np.arange(1, n_tot + 1)
    a = (n_tot + 1) / 2.0 - np.abs(r - (n_tot + 1) / 2.0)
    return _rank_test(xa, ya, a)


def mood(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Mood (1954) squared-centered-rank test."""
    xa, ya = _check2(x, y)
    n_tot = xa.size + ya.size
    r = np.arange(1, n_tot + 1)
    a = np.asarray((r - (n_tot + 1) / 2) ** 2, dtype=np.float64)
    return _rank_test(xa, ya, a)


def klotz(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Klotz (1962) normal-scores-squared scale test."""
    xa, ya = _check2(x, y)
    n_tot = xa.size + ya.size
    i = np.arange(1, n_tot + 1)
    a = stats.norm.ppf(i / (n_tot + 1)) ** 2
    return _rank_test(xa, ya, a)


def capon(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Capon (1961) squared-normal-scores (== Klotz locally)."""
    return klotz(x, y)


def conover(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Conover (1980) squared-rank scale test.

    Rank |x - joint median| across the pooled sample and
    use squared ranks as scores — the classical
    nonparametric dispersion benchmark."""
    xa, ya = _check2(x, y)
    n, m = xa.size, ya.size
    n_tot = n + m
    med = np.median(np.concatenate([xa, ya]))
    dev = np.concatenate([np.abs(xa - med), np.abs(ya - med)])
    a = stats.rankdata(dev) ** 2
    stat = float(a[:n].sum())
    e = n * float(a.mean())
    var = n * m / (n_tot * (n_tot - 1)) * float(((a - a.mean()) ** 2).sum())
    z = (stat - e) / np.sqrt(var)
    return {
        "stat": stat,
        "z": float(z),
        "pvalue": float(2 * stats.norm.sf(abs(z))),
    }


def gastwirth(x: FloatArray, y: FloatArray, p: float = 0.5) -> dict[str, float]:
    """Gastwirth (1965) winsorized percentile score test."""
    xa, ya = _check2(x, y)
    n_tot = xa.size + ya.size
    i = np.arange(1, n_tot + 1) / (n_tot + 1)
    a = np.clip(i - (1 - p), 0, None) ** 2 + np.clip(p - i, 0, None) ** 2
    return _rank_test(xa, ya, a)


def bench_score_scale(seed: int = 525) -> dict[str, float]:
    """SYNTHETIC: N(0,1) vs N(0,2): every rank-scale test
    must detect; equal scale must not be over-rejected."""
    rng = np.random.default_rng(seed)
    R = 60
    tests = {
        "ab": ansari_bradley,
        "mood": mood,
        "klotz": klotz,
        "conover": conover,
        "gast": gastwirth,
    }
    rej_null = {k: 0 for k in tests}
    rej_alt = {k: 0 for k in tests}
    for _ in range(R):
        x1 = rng.normal(0, 1, 50)
        y1 = rng.normal(0, 1, 50)
        x2 = rng.normal(0, 1, 50)
        y2 = rng.normal(0, 2.0, 50)
        for name, fn in tests.items():
            rej_null[name] += int(fn(x1, y1)["pvalue"] < 0.05)
            rej_alt[name] += int(fn(x2, y2)["pvalue"] < 0.05)
    out: dict[str, float] = {}
    for name in tests:
        out[f"synthetic_type1_{name}"] = rej_null[name] / R
        out[f"synthetic_power_{name}"] = rej_alt[name] / R
        if rej_alt[name] < R * 0.8:
            raise ValueError(f"{name} misses 2x scale shift")
        if rej_null[name] > R * 0.18:
            raise ValueError(f"{name} over-rejects")
    return out
