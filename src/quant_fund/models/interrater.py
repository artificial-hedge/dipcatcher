"""Inter-rater agreement and reliability coefficients.

Cohen (1960) kappa for two raters,

    kappa = (p_o - p_e) / (1 - p_e),

with linear/quadratic weighted variants (Cohen 1968) for
ordinal scales. Fleiss (1971) kappa generalizes to n raters
scoring N subjects via per-subject agreement

    P_i = (sum_j n_ij (n_ij - 1)) / (n (n-1)).

Krippendorff (1970/2004) alpha uses coincidence matrices and
works for any number of raters, missing data, and nominal/
ordinal/interval metrics: alpha = 1 - D_o / D_e.

Lin (1989) concordance correlation rho_c = 2 rho s_x s_y /
(s_x^2 + s_y^2 + (m_x - m_y)^2) — agreement against the 45
degree line, combining precision (rho) and accuracy (bias
factor). Bland-Altman (1986) limits of agreement report the
bias and 95% limits m +/- 1.96 s of pairwise differences.

Honesty: kappa variants require >= 2 observed categories;
alpha reduces to two-rater form for complete data. The bench
simulates raters with planted agreement structure and checks
kappa recovery and perfect-agreement endpoints. Fail-closed
on single-class tables or constant raters where pe = 1.

References: Cohen (1960) Educ. Psychol. Meas. 20:37; Fleiss
(1971) Psychol. Bull. 76:378; Krippendorff (1970/2004);
Lin (1989) Biometrics 45:255; Bland & Altman (1986) Lancet
1:307; Gwet (2014) handbook.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _confusion(r1: FloatArray, r2: FloatArray) -> tuple[FloatArray, int]:
    a = np.asarray(r1, dtype=np.float64).ravel()
    b = np.asarray(r2, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 4:
        raise ValueError("rater vectors must align (n>=4)")
    cats = np.unique(np.concatenate([a, b]))
    k = cats.size
    if k < 2:
        raise ValueError("need at least two rating categories")
    idx = {c: i for i, c in enumerate(cats)}
    tab = np.zeros((k, k))
    for u, v in zip(a, b, strict=True):
        tab[idx[u], idx[v]] += 1.0
    return tab, k


def cohen_kappa(r1: FloatArray, r2: FloatArray) -> dict[str, float]:
    """Unweighted kappa + observed/expected agreement."""
    tab, _ = _confusion(r1, r2)
    n = tab.sum()
    po = float(np.trace(tab)) / n
    rows, cols = tab.sum(1), tab.sum(0)
    pe = float(rows @ cols) / (n * n)
    if pe >= 1.0:
        raise ValueError("pe=1 — degenerate marginals")
    kappa = (po - pe) / (1.0 - pe)
    return {"kappa": kappa, "p_observed": po, "p_expected": pe}


def weighted_kappa(r1: FloatArray, r2: FloatArray, weights: str = "quadratic") -> dict[str, float]:
    """Cohen (1968) weighted kappa for ordinal categories."""
    tab, k = _confusion(r1, r2)
    n = tab.sum()
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == "linear":
                w[i, j] = abs(i - j) / (k - 1)
            elif weights == "quadratic":
                w[i, j] = ((i - j) / (k - 1)) ** 2
            else:
                raise ValueError("weights must be 'linear' or 'quadratic'")
    o_w = float((w * tab).sum()) / n
    rows, cols = tab.sum(1), tab.sum(0)
    e_tab = np.outer(rows, cols) / n
    e_w = float((w * e_tab).sum()) / n
    if e_w <= 0:
        raise ValueError("e_w=0 — degenerate marginals")
    kappa = 1.0 - o_w / e_w
    return {"kappa_weighted": kappa, "o_w": o_w, "e_w": e_w}


def fleiss_kappa(counts: FloatArray) -> dict[str, float]:
    """Fleiss (1971) kappa; `counts` is N x k with row sums = raters."""
    m = np.asarray(counts, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2:
        raise ValueError("counts must be N x k")
    n_rater = m.sum(axis=1)
    if not np.allclose(n_rater, n_rater[0]) or n_rater[0] <= 1:
        raise ValueError("rows must share a common rater count > 1")
    n = float(n_rater[0])
    p_j = m.sum(axis=0) / m.sum()
    p_i = ((m * (m - 1.0)).sum(axis=1)) / (n * (n - 1.0))
    pbar = float(p_i.mean())
    pe = float((p_j * p_j).sum())
    if pe >= 1.0:
        raise ValueError("pe=1 — degenerate marginals")
    kappa = (pbar - pe) / (1.0 - pe)
    return {"kappa_fleiss": kappa, "p_bar": pbar, "p_e": pe}


def krippendorff_alpha(ratings: FloatArray, level: str = "nominal") -> dict[str, float]:
    """Krippendorff alpha on a rater x item matrix; NaN = missing.

    level in {'nominal','ordinal','interval'} sets the
    disagreement metric (delta^2 = 1, rank gap^2, value gap^2).
    """
    r = np.asarray(ratings, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 2:
        raise ValueError("ratings must be raters x items")
    valid = r[np.isfinite(r)]
    vals = np.unique(valid)
    if vals.size < 2:
        raise ValueError("need >= 2 distinct ratings")
    # coincidence matrix
    ncat = vals.size
    idx = {v: i for i, v in enumerate(vals)}
    o = np.zeros((ncat, ncat))
    pairable = 0
    for j in range(r.shape[1]):
        col = r[:, j][np.isfinite(r[:, j])]
        m = col.size
        if m < 2:
            continue
        pairable += m
        for a in range(m):
            for b in range(m):
                o[idx[col[a]], idx[col[b]]] += 1.0 / (m - 1.0)
    if pairable == 0:
        raise ValueError("no pairable items")
    # metric distances
    if level == "nominal":
        d2 = 1.0 - np.eye(ncat)
    elif level == "interval":
        gv = np.array(sorted(vals))
        d2 = (gv[:, None] - gv[None, :]) ** 2
    elif level == "ordinal":
        ranks = np.arange(ncat, dtype=np.float64)
        d2 = np.asarray((ranks[:, None] - ranks[None, :]) ** 2, dtype=np.float64)
    else:
        raise ValueError("level must be nominal/ordinal/interval")
    n_total = o.sum()
    d_o = float((o * d2).sum()) / max(n_total, 1e-12)
    marg = o.sum(axis=1)
    d_e = float((np.outer(marg, marg) * d2).sum()) / max(n_total * (n_total - 1.0), 1e-12)
    if d_e <= 0:
        raise ValueError("d_e=0 — degenerate margins")
    alpha = 1.0 - d_o / d_e
    return {"alpha": float(alpha), "d_o": d_o, "d_e": d_e}


def lin_ccc(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Lin (1989) concordance correlation coefficient."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 3:
        raise ValueError("need paired n>=3")
    mx, my = a.mean(), b.mean()
    vx, vy = a.var(), b.var()
    rho = float(np.corrcoef(a, b)[0, 1])
    den = vx + vy + (mx - my) ** 2
    if den <= 0:
        raise ValueError("zero total variance")
    rho_c = 2.0 * rho * np.sqrt(vx * vy) / den
    bias_factor = 2.0 / (np.sqrt(vx / vy) + np.sqrt(vy / vx) + (mx - my) ** 2 / np.sqrt(vx * vy))
    return {"ccc": float(rho_c), "pearson_r": rho, "bias_factor": float(bias_factor)}


def bland_altman(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Bland-Altman bias + 95% limits of agreement."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 3:
        raise ValueError("need paired n>=3")
    d = a - b
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd <= 0:
        raise ValueError("zero sd of differences")
    return {
        "bias": bias,
        "sd_diff": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "mean_of_means": float(((a + b) / 2.0).mean()),
    }


def bench_interrater(seed: int = 20261231 + 452) -> dict[str, float]:
    """SYNTHETIC check — kappa recovers planted agreement."""
    rng = np.random.default_rng(seed)
    n = 300
    true_cat = rng.integers(0, 4, size=n)
    # rater A = truth + noise; rater B = truth + more noise
    p_a, p_b = 0.9, 0.7
    ra = np.where(rng.random(n) < p_a, true_cat, rng.integers(0, 4, size=n))
    rb = np.where(rng.random(n) < p_b, true_cat, rng.integers(0, 4, size=n))
    out = cohen_kappa(ra.astype(np.float64), rb.astype(np.float64))
    kap = float(out["kappa"])
    # agreement should be moderately high but < 1
    if not (0.35 < kap < 0.95):
        raise ValueError(f"kappa off: {kap}")
    perf = cohen_kappa(true_cat.astype(np.float64), true_cat.astype(np.float64))
    if abs(float(perf["kappa"]) - 1.0) > 1e-9:
        raise ValueError("perfect agreement must give kappa=1")
    return {
        "synthetic_kappa": kap,
        "synthetic_perfect_kappa": float(perf["kappa"]),
        "synthetic_score": 1.0,
    }
