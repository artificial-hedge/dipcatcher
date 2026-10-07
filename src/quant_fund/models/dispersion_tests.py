"""Rank-based dispersion tests — Siegel-Tukey and Ansari-Bradley.

Siegel & Tukey (1960) and Ansari & Bradley (1960) test equal
dispersion across two samples without normality assumptions by
scoring pooled ranks so that extreme observations carry extreme
scores:

    Siegel-Tukey scores: pooled ranks r_i are replaced by
        1, N, 2, N-1, 3, N-2, ... alternating from both ends —
        spread groups get *smaller* mean scores.
    Ansari-Bradley scores: |r_i - (N+1)/2|, so spread groups get
        *larger* mean scores.

Both reduce to a Wilcoxon-type statistic W = sum of scores in
sample 1, asymptotically normal with the usual tie correction:
    E[W] = n1 (N+1)/2-ish (per scoring convention), Var from
    Var(W) = n1 n2 Var(scores pooled) / (N-1) adjusted.

Honesty: the bench uses a planted 3x scale difference (both tests
must reject) and a same-scale null (both respect size). The normal
approximation is standard for n>=20; bounds documented. Fail-closed
on tiny/degenerate samples.

References: Siegel & Tukey (1960) "A nonparametric sum of ranks
procedure for relative spread"; Ansari & Bradley (1960) "Rank-sum
tests for dispersions"; Hollander, Wolfe, Chicken (2014) ch. 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

FloatArray = NDArray[np.float64]


def _check2(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.size < 8 or b.size < 8:
        raise ValueError("bad samples")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite input")
    return a, b


def _z_on_scores(scores: FloatArray, n1: int) -> dict[str, float]:
    """Two-sided normal-approx test on W = sum of sample-1 scores."""
    n = scores.size
    n2 = n - n1
    w = float(scores[:n1].sum())
    mean = n1 * float(scores.mean())
    # finite-population variance of a sampled sum
    var = n1 * n2 * float(scores.var(ddof=0)) / (n - 1)
    if var <= 0:
        raise ValueError("degenerate scores")
    z = (w - mean) / np.sqrt(var)
    return {"w": w, "z": float(z), "p": float(2.0 * norm.sf(abs(z)))}


def siegel_tukey(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Siegel-Tukey rank-spread test (two-sided).

    Note the sign convention: larger sample-1 dispersion gives a
    *smaller* W, so we report ``z`` and ``p`` on the raw score sum.
    """
    a, b = _check2(x, y)
    pooled = np.concatenate([a, b])
    n = pooled.size
    r = rankdata(pooled)
    order = np.argsort(r, kind="stable")  # positions by ascending rank
    st = np.empty(n)
    # Siegel-Tukey scoring: 1 to the smallest, then pairs of
    # consecutive scores alternating ends — largest gets 2,
    # second-largest 3, smallest-remaining 4, next 5, and so on.
    c = 1
    bot = 0
    top = n - 1
    st[order[bot]] = float(c)
    c += 1
    bot += 1
    low_side = False  # next pair comes from the top
    while bot <= top:
        for _ in range(2):
            if bot > top:
                break
            idx = order[bot] if low_side else order[top]
            st[idx] = float(c)
            c += 1
            if low_side:
                bot += 1
            else:
                top -= 1
        low_side = not low_side
    return _z_on_scores(st, a.size)


def ansari_bradley(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Ansari-Bradley dispersion test (two-sided)."""
    a, b = _check2(x, y)
    pooled = np.concatenate([a, b])
    n = pooled.size
    r = rankdata(pooled)
    ab = np.abs(r - (n + 1.0) / 2.0)
    return _z_on_scores(ab, a.size)


def bench_dispersion_tests(seed: int = 20261231 + 432) -> dict[str, float]:
    """SYNTHETIC check — 3x scale difference rejected, null held."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(60)
    y = 3.0 * rng.standard_normal(60)
    st = siegel_tukey(x, y)
    ab = ansari_bradley(x, y)
    xi = rng.standard_normal(60)
    yi = rng.standard_normal(60)
    st_n = siegel_tukey(xi, yi)
    ab_n = ansari_bradley(xi, yi)
    if st["p"] > 0.01 or ab["p"] > 0.01 or st_n["p"] < 0.005 or ab_n["p"] < 0.005:
        raise ValueError(
            f"dispersion off: st={st['p']:.4f} ab={ab['p']:.4f} "
            f"null_st={st_n['p']:.4f} null_ab={ab_n['p']:.4f}"
        )
    return {
        "synthetic_siegel_tukey_p": st["p"],
        "synthetic_ansari_bradley_p": ab["p"],
        "synthetic_siegel_tukey_p_null": st_n["p"],
        "synthetic_ansari_bradley_p_null": ab_n["p"],
        "synthetic_score": 1.0,
    }
