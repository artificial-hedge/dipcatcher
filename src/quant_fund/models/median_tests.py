"""Mood's median test and pairwise median contrasts.

Mood (1950): for k independent samples, count observations above
the grand median in each group; under H0 (common median) the
above/below counts are homogeneous, giving a Pearson chi^2 on the
k x 2 table:

    X2 = sum_g (O_g - E_g)^2 / E_g  ~ chi^2_{k-1}

For exactly two samples the test is a difference-in-medians analog
of Mann-Whitney; Brown & Mood (1951) give the linear-combination
form.

Honesty: the bench plants a +1.5 median shift in one of three
groups (rejected) and a null (held). Chi^2 approximation requires
expected counts >= ~5; documented. Fail-closed on fewer than two
groups or a degenerate 2x2 table (all on one side of the median —
a valid outcome only when medians truly agree at discrete mass).

References: Mood (1950) "Introduction to the Theory of
Statistics" ch. 16; Brown & Mood (1951) "On median tests for
linear hypotheses"; Hollander, Wolfe, Chicken (2014) ch. 6.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

FloatArray = NDArray[np.float64]


def _check_groups(groups: tuple[FloatArray, ...]) -> list[FloatArray]:
    gs = [np.asarray(g, dtype=float) for g in groups]
    if len(gs) < 2:
        raise ValueError("need >=2 groups")
    for g in gs:
        if g.ndim != 1 or g.size < 5 or not np.isfinite(g).all():
            raise ValueError("bad group")
    return gs


def mood_median(*groups: FloatArray) -> dict[str, float]:
    """Mood's chi^2 median-homogeneity test across >=2 groups."""
    gs = _check_groups(groups)
    pooled = np.concatenate(gs)
    med = float(np.median(pooled))
    k = len(gs)
    above = np.array([float((g > med).sum()) for g in gs])
    below = np.array([float((g < med).sum()) for g in gs])
    ns = np.array([float(g.size) for g in gs])
    # ties at the median are dropped from both rows' counts
    tab = np.stack([above, below], axis=1)
    total_above = tab[:, 0].sum()
    total_below = tab[:, 1].sum()
    if total_above <= 0 or total_below <= 0:
        raise ValueError("degenerate table")
    exp = np.stack([ns * total_above / ns.sum(), ns * total_below / ns.sum()], axis=1)
    x2 = float(((tab - exp) ** 2 / exp).sum())
    p = float(chi2.sf(x2, k - 1))
    return {"chi2": x2, "p": p, "grand_median": med}


def pairwise_median_contrast(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Two-sample Mood test plus a normal-approx z on the count
    difference (Brown-Mood sign-count form)."""
    gs = _check_groups((x, y))
    out = mood_median(*gs)
    med = out["grand_median"]
    a1 = float((gs[0] > med).sum())
    b1 = float((gs[1] > med).sum())
    n1_eff = a1 + float((gs[0] < med).sum())
    n2_eff = b1 + float((gs[1] < med).sum())
    phat = (a1 + b1) / (n1_eff + n2_eff)
    se = np.sqrt(phat * (1 - phat) * (1 / n1_eff + 1 / n2_eff))
    z = (a1 / n1_eff - b1 / n2_eff) / max(se, 1e-12)
    out["z"] = float(z)
    out["p_two_sample"] = float(2 * norm.sf(abs(z)))
    return out


def bench_median_tests(seed: int = 20261231 + 433) -> dict[str, float]:
    """SYNTHETIC check — planted shift rejected, null held."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(50)
    b = rng.standard_normal(50)
    c = rng.standard_normal(50) + 1.5
    out = mood_median(a, b, c)
    out_n = mood_median(rng.standard_normal(50), rng.standard_normal(50), rng.standard_normal(50))
    pair = pairwise_median_contrast(rng.standard_normal(60), rng.standard_normal(60) + 1.6)
    if out["p"] > 0.01 or out_n["p"] < 0.005 or pair["p_two_sample"] > 0.01:
        raise ValueError(
            f"median off: p={out['p']:.4f} null={out_n['p']:.4f} pair={pair['p_two_sample']:.4f}"
        )
    return {
        "synthetic_mood_p": out["p"],
        "synthetic_mood_p_null": out_n["p"],
        "synthetic_pair_median_p": pair["p_two_sample"],
        "score": 1.0,
    }
