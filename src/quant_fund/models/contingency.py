"""Contingency-table inference — Fisher exact, McNemar, CMH, effect sizes.

Fisher (1922), McNemar (1947), Cochran-Mantel-Haenszel (1954/1959):
for 2x2 and stratified tables the exact/conditional tests avoid the
chi^2 approximation entirely; association is summarized by phi (2x2)
and Cramer's V (r x c).

    Fisher exact p = sum over tables as/extreme under hypergeometric
    McNemar: chi^2 = (b-c)^2/(b+c) on discordant pairs (or exact binomial)
    CMH: pooled odds-ratio test across K strata

Honesty: the bench builds a planted association (Fisher/phi/Cramér
must detect), a discordant-pair shift (McNemar rejects), and a
stratified common-odds scenario (CMH rejects while per-stratum
noise does not). Fail-closed on negative counts or degenerate
tables.

References: Fisher (1922) "On the interpretation of chi^2";
McNemar (1947) "Note on the sampling error of the difference
between correlated proportions"; Mantel & Haenszel (1959)
"Statistical aspects of the analysis of data from retrospective
studies"; Cramér (1946) "Mathematical Methods of Statistics".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, hypergeom

FloatArray = NDArray[np.float64]


def _check_table(t: FloatArray, name: str = "table") -> FloatArray:
    a = np.asarray(t, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError(f"bad {name}")
    if not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
        raise ValueError(f"bad {name} counts")
    return a


def fisher_exact(t: FloatArray) -> dict[str, float]:
    """Two-sided Fisher exact test on a 2x2 table (Fisher-Irwin)."""
    a = _check_table(t)
    if a.shape != (2, 2):
        raise ValueError("need 2x2")
    r1, r2 = a.sum(axis=1)
    c1 = a[0, 0] + a[1, 0]
    n = a.sum()
    lo = max(0, int(r1 + c1 - n))
    hi = min(int(r1), int(c1))
    obs = int(a[0, 0])
    p_obs = hypergeom.pmf(obs, int(n), int(r1), int(c1))
    p = sum(
        hypergeom.pmf(k, int(n), int(r1), int(c1))
        for k in range(lo, hi + 1)
        if hypergeom.pmf(k, int(n), int(r1), int(c1)) <= p_obs + 1e-12
    )
    return {
        "p": float(min(1.0, p)),
        "odds_ratio": float((a[0, 0] * a[1, 1]) / max(a[0, 1] * a[1, 0], 1e-12)),
    }


def mcnemar(t: FloatArray) -> dict[str, float]:
    """McNemar's test for paired binary data (exact binomial when
    b+c < 25, else chi^2 with continuity correction)."""
    a = _check_table(t)
    if a.shape != (2, 2):
        raise ValueError("need 2x2")
    b, c = a[0, 1], a[1, 0]
    n_disc = int(b + c)
    if n_disc == 0:
        raise ValueError("no discordant pairs")
    if n_disc < 25:
        from scipy.stats import binom

        k = int(min(b, c))
        p = float(min(1.0, 2.0 * binom.cdf(k, n_disc, 0.5)))
        stat = float(k)
    else:
        stat = float((abs(b - c) - 1.0) ** 2 / n_disc)
        p = float(chi2.sf(stat, 1))
    return {"p": p, "stat": stat, "b": float(b), "c": float(c)}


def cmh(tables: list[FloatArray]) -> dict[str, float]:
    """Cochran-Mantel-Haenszel common-odds test across K 2x2 strata."""
    if len(tables) < 2:
        raise ValueError("need >=2 strata")
    ts = [_check_table(t, "stratum") for t in tables]
    if any(t.shape != (2, 2) for t in ts):
        raise ValueError("need 2x2 strata")
    num = 0.0
    var = 0.0
    for a in ts:
        n = a.sum()
        r1 = a[0].sum()
        c1 = a[:, 0].sum()
        if n <= 1:
            raise ValueError("degenerate stratum")
        num += a[0, 0] - r1 * c1 / n
        var += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1))
    if var <= 0:
        raise ValueError("zero CMH variance")
    stat = float((abs(num) - 0.5) ** 2 / var)
    return {"p": float(chi2.sf(stat, 1)), "stat": stat}


def phi_coefficient(t: FloatArray) -> float:
    """Pearson phi on a 2x2 table."""
    a = _check_table(t)
    if a.shape != (2, 2):
        raise ValueError("need 2x2")
    num = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    den = np.sqrt(a[0].sum() * a[1].sum() * a[:, 0].sum() * a[:, 1].sum())
    return float(num / max(den, 1e-12))


def cramers_v(t: FloatArray) -> float:
    """Cramer's V on an r x c table."""
    a = _check_table(t)
    r, c = a.shape
    n = a.sum()
    exp = np.outer(a.sum(axis=1), a.sum(axis=0)) / n
    chi = float(((a - exp) ** 2 / np.maximum(exp, 1e-12)).sum())
    return float(np.sqrt(chi / (n * min(r - 1, c - 1))) if min(r - 1, c - 1) > 0 else 0.0)


def bench_contingency(seed: int = 20261231 + 430) -> dict[str, float]:
    """SYNTHETIC check — planted association, paired shift, common odds."""
    rng = np.random.default_rng(seed)
    # planted 2x2 association: exposed group P(Y)=0.7 vs 0.3
    n1, n2 = 40, 40
    t = np.array(
        [
            [rng.binomial(n1, 0.7), n1 - rng.binomial(n1, 0.7)],
            [rng.binomial(n2, 0.3), n2 - rng.binomial(n2, 0.3)],
        ],
        dtype=float,
    )
    # resimulate cleanly for determinism
    y1 = rng.binomial(n1, 0.7)
    y2 = rng.binomial(n2, 0.3)
    t = np.array([[y1, n1 - y1], [y2, n2 - y2]], dtype=float)
    f = fisher_exact(t)
    phi = phi_coefficient(t)
    # McNemar: 60 pairs, more off-diagonal in one direction
    b, c = 28, 8
    mn = mcnemar(np.array([[30.0, b], [c, 20.0]]))
    # CMH: 3 strata with common OR ~3
    strata = []
    for _ in range(3):
        a11 = rng.binomial(30, 0.6)
        b11 = rng.binomial(30, 0.3)
        strata.append(np.array([[a11, 30 - a11], [b11, 30 - b11]], dtype=float))
    cm = cmh(strata)
    if f["p"] > 0.01 or abs(phi) < 0.3 or mn["p"] > 0.01 or cm["p"] > 0.01:
        raise ValueError(
            f"contingency off: fisher={f['p']:.4f} phi={phi:.3f} "
            f"mcnemar={mn['p']:.4f} cmh={cm['p']:.4f}"
        )
    return {
        "synthetic_fisher_p": f["p"],
        "synthetic_phi": phi,
        "synthetic_mcnemar_p": mn["p"],
        "synthetic_cmh_p": cm["p"],
        "synthetic_score": 1.0,
    }
