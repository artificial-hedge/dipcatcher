"""Scale/spread homogeneity tests across k groups (SYNTHETIC).

Canonical references:

- Levene (1960) 'Robust tests for equality of
  variances' in Olkin ed. — ANOVA on |x - group mean|.
- Brown & Forsythe (1974) 'Robust tests for the
  equality of variances' JASA 69 — same with the group
  median (robust to non-normality).
- Fligner & Killeen (1976) 'Distribution-free two-sample
  tests for scale' JASA 71 — rank test on
  |x - joint median| with normal-scores weights; the
  most robust of the classical set under heavy tails.
- O'Brien (1979) 'A general ANOVA method for robust
  tests of additive models for variances' JASA 74 —
  transformed cell values
  r_i = ((w+n_i-2)n_i(x_i-mean)^2 - w(n_i-1)s_i^2) /
        ((n_i-1)(n_i-2)), w=0.5,
  then ordinary ANOVA F.

`bench_scale`: equal-variance groups accepted ~95%;
doubling one group's scale is rejected by every test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _check_groups(groups: list[FloatArray]) -> list[FloatArray]:
    gs = []
    for g in groups:
        ga = np.asarray(g, dtype=np.float64).ravel()
        if ga.size < 4 or not np.isfinite(ga).all():
            raise ValueError("bad group")
        gs.append(ga)
    if len(gs) < 2:
        raise ValueError("need >=2 groups")
    return gs


def _anova_f(vals: FloatArray, labels: FloatArray) -> dict[str, float]:
    """One-way ANOVA F on transformed values."""
    k = len(np.unique(labels))
    n = vals.size
    grand = float(vals.mean())
    ss_b = 0.0
    ss_w = 0.0
    for lab in np.unique(labels):
        v = vals[labels == lab]
        ss_b += v.size * (v.mean() - grand) ** 2
        ss_w += float(((v - v.mean()) ** 2).sum())
    f = (ss_b / (k - 1)) / (ss_w / (n - k))
    return {"stat": float(f), "pvalue": float(stats.f.sf(f, k - 1, n - k))}


def levene(groups: list[FloatArray], center: str = "mean") -> dict[str, float]:
    """Levene (1960) / Brown-Forsythe (1974) test.

    center='mean' is Levene's original; 'median' is the
    robust Brown-Forsythe variant."""
    gs = _check_groups(groups)
    vals = np.concatenate(gs)
    labels = np.repeat(np.arange(len(gs)), [g.size for g in gs])
    z = np.zeros(vals.size)
    for i, g in enumerate(gs):
        c = float(np.median(g)) if center == "median" else float(g.mean())
        z[labels == i] = np.abs(g - c)
    out = _anova_f(z, labels.astype(np.float64))
    return {"stat": out["stat"], "pvalue": out["pvalue"]}


def fligner_killeen(groups: list[FloatArray]) -> dict[str, float]:
    """Fligner-Killeen (1976) chi-square rank scale test."""
    gs = _check_groups(groups)
    n_tot = sum(g.size for g in gs)
    z = np.concatenate([np.abs(g - np.median(g)) for g in gs])
    labels = np.repeat(np.arange(len(gs)), [g.size for g in gs])
    # normal scores of the pooled |deviation| ranks
    ranks = stats.rankdata(z)
    a = stats.norm.ppf((ranks - 0.5) / n_tot)
    a = np.where(np.isfinite(a), a, 0.0)
    a2_bar = float((a**2).mean())
    num = 0.0
    for i, g in enumerate(gs):
        num += g.size * (a[labels == i].mean() ** 2)
    stat = num / a2_bar
    return {
        "stat": float(stat),
        "pvalue": float(stats.chi2.sf(stat, len(gs) - 1)),
    }


def obrien(groups: list[FloatArray], w: float = 0.5) -> dict[str, float]:
    """O'Brien (1979) transformed-value ANOVA."""
    gs = _check_groups(groups)
    labels = np.repeat(np.arange(len(gs)), [g.size for g in gs])
    vals = np.concatenate(gs)
    r = np.zeros(vals.size)
    for i, g in enumerate(gs):
        ni = g.size
        if ni < 3:
            raise ValueError("O'Brien needs n_i >= 3")
        m = g.mean()
        s2 = float(g.var(ddof=1))
        mask = labels == i
        r[mask] = ((w + ni - 2) * ni * (g - m) ** 2 - w * (ni - 1) * s2) / ((ni - 1) * (ni - 2))
    out = _anova_f(r, labels.astype(np.float64))
    return {"stat": out["stat"], "pvalue": out["pvalue"], "w": float(w)}


def bench_scale(seed: int = 524) -> dict[str, float]:
    """SYNTHETIC: 3 groups, N(0,1)/N(0,1)/N(0,2.2);
    all four tests must detect while keeping size on
    equal-variance replicates."""
    rng = np.random.default_rng(seed)
    R = 60
    tests = {
        "levene": lambda g: levene(g, "mean"),
        "bf": lambda g: levene(g, "median"),
        "fk": fligner_killeen,
        "ob": obrien,
    }
    rej_null = {k: 0 for k in tests}
    rej_alt = {k: 0 for k in tests}
    for _ in range(R):
        same = [rng.normal(0, 1, 60), rng.normal(0, 1, 60), rng.normal(0, 1, 60)]
        diff = [rng.normal(0, 1, 60), rng.normal(0, 1, 60), rng.normal(0, 2.2, 60)]
        for name, fn in tests.items():
            rej_null[name] += int(fn(same)["pvalue"] < 0.05)
            rej_alt[name] += int(fn(diff)["pvalue"] < 0.05)
    out: dict[str, float] = {}
    for name in tests:
        out[f"synthetic_type1_{name}"] = rej_null[name] / R
        out[f"synthetic_power_{name}"] = rej_alt[name] / R
        if rej_alt[name] < R * 0.9:
            raise ValueError(f"{name} misses scale shift")
        if rej_null[name] > R * 0.15:
            raise ValueError(f"{name} over-rejects")
    return out
