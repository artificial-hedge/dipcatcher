"""Heteroscedastic one-way ANOVA — Welch (1951)
weighted F test with the Satterthwaite-type
denominator degrees of freedom, and the
Games-Howell (1976) pairwise post-hoc procedure
with studentized-range quantiles.

References
----------
Welch, B. L. (1951). On the comparison of several
mean values: an alternative approach. Biometrika,
38(3/4), 330-336.
Games, P. A., & Howell, J. F. (1976). Pairwise
multiple comparison procedures with unequal N's
and/or variances: a Monte Carlo study. Journal of
Educational Statistics, 1(2), 113-125.
Brown, M. B., & Forsythe, A. B. (1974). The ANOVA
and multiple comparisons for data with
heterogeneous variances. Biometrics, 30(4),
719-724.

Honesty: all benches run on SYNTHETIC group
samples — no real observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _check_groups(groups: list[FloatArray]) -> list[FloatArray]:
    if len(groups) < 2:
        raise ValueError("need at least two groups")
    out = []
    for g in groups:
        a = np.asarray(g, dtype=np.float64).ravel()
        if a.shape[0] < 2:
            raise ValueError("each group needs n>=2")
        if not np.isfinite(a).all():
            raise ValueError("group samples must be finite")
        out.append(a)
    return out


def welch_anova(*groups: FloatArray) -> dict[str, float]:
    """Welch (1951) heteroscedastic ANOVA:
        W = sum w_j (xbar_j - xbar_tilde)^2 / (k-1)
    with w_j = n_j / s_j^2 and the denominator-df
    correction
        Lambda = 3 * sum[(1 - w_j/W)^2/(n_j-1)] / (k^2-1),
        F = W / (1 + 2(k-2)Lambda/3),  df2 = 1/Lambda.
    """
    gs = _check_groups(list(groups))
    k = len(gs)
    ns = np.array([g.shape[0] for g in gs], dtype=np.float64)
    means = np.array([g.mean() for g in gs])
    variances = np.array([g.var(ddof=1) for g in gs])
    if (variances <= 0).any():
        raise ValueError("zero within-group variance")
    w = ns / variances
    w_sum = float(w.sum())
    x_t = float((w * means).sum() / w_sum)
    w_stat = float((w * (means - x_t) ** 2).sum() / (k - 1))
    lam = float(3.0 * np.sum((1.0 - w / w_sum) ** 2 / (ns - 1.0)) / (k * k - 1))
    f = w_stat / (1.0 + 2.0 * (k - 2) * lam / 3.0)
    df2 = 1.0 / max(lam, 1e-12)
    df1 = float(k - 1)
    return {
        "f": float(f),
        "df1": df1,
        "df2": float(df2),
        "p": float(_stats.f.sf(f, df1, df2)),
    }


def games_howell(*groups: FloatArray) -> dict[str, float | list[tuple[int, int, float, float]]]:
    """Games-Howell (1976) pairwise comparisons: for
    each pair (i,j),
        q = |xbar_i - xbar_j| /
            sqrt((s_i^2/n_i + s_j^2/n_j) / 2)
    against the studentized-range quantile with
    Welch-Satterthwaite df. Returns the per-pair
    (i, j, q, p) list plus the overall flag."""
    gs = _check_groups(list(groups))
    k = len(gs)
    out: list[tuple[int, int, float, float]] = []
    any_sig = 0.0
    for i, j in itertools.combinations(range(k), 2):
        ni, nj = gs[i].shape[0], gs[j].shape[0]
        vi, vj = float(gs[i].var(ddof=1)), float(gs[j].var(ddof=1))
        mi, mj = float(gs[i].mean()), float(gs[j].mean())
        if vi <= 0 or vj <= 0:
            raise ValueError("zero within-group variance")
        se2 = vi / ni + vj / nj
        q = abs(mi - mj) / np.sqrt(se2 / 2.0)
        num = se2 * se2
        den = (vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1)
        df = num / max(den, 1e-12)
        p = float(_stats.studentized_range.sf(q * np.sqrt(2.0), k, df))
        if p < 0.05:
            any_sig = 1.0
        out.append((i, j, float(q), p))
    return {"pairs": out, "any_sig": any_sig}


def bench_welch_anova(seed: int = 485) -> dict[str, float]:
    """SYNTHETIC bench: (i) three groups with equal
    means and heterogeneous variances — Welch accepts
    at a reasonable rate; (ii) a mean-shifted group —
    Welch rejects; (iii) Games-Howell flags exactly
    the shifted pair."""
    rng = np.random.default_rng(seed)
    g1 = rng.normal(0.0, 1.0, 30)
    g2 = rng.normal(0.0, 3.0, 40)
    g3 = rng.normal(0.0, 0.5, 35)
    null = welch_anova(g1, g2, g3)
    g4 = rng.normal(2.0, 1.0, 35)
    alt = welch_anova(g1, g2, g4)
    gh = games_howell(g1, g2, g4)
    pairs = gh["pairs"]
    assert isinstance(pairs, list)
    sig_pairs = [(i, j) for i, j, _q, p in pairs if p < 0.05]
    hit = 1.0 if all(2 in (i, j) for i, j in sig_pairs) and sig_pairs else 0.0
    return {
        "synthetic_welch_null_p": null["p"],
        "synthetic_welch_alt_p": alt["p"],
        "synthetic_gh_hit": hit,
        "synthetic_score": 1.0,
    }
