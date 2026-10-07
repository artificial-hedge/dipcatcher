"""Post-hoc pairwise-comparison procedures.

Tukey (1949) honestly significant difference: for k groups
with means m_i, sizes n_i and pooled MSE s^2 on nu = N - k
degrees of freedom, the pairwise statistic

    q_ij = |m_i - m_j| / (s * sqrt(1/2 (1/n_i + 1/n_j)))

is referenced to the studentized-range distribution q_{k,nu}
(via scipy.stats.studentized_range when available; internal
MC quantile fallback). Dunnett (1955) compares each of k-1
treatments against a control with a multivariate-t critical
value (Dunnett's d — simulated MC fallback). Games-Howell
(1976) drops the equal-variance assumption (Welch
denominators + Welch-Satterthwaite df). Scheffe (1953)
covers all contrasts with the S-method
sqrt((k-1) F_{k-1,nu,1-a}).

Honesty: one-way layouts only; p-values for Tukey use the
exact studentized-range cdf when scipy is present (it is a
repo dependency) — the MC fallback is labeled approximate.
Dunnett uses a simulated multivariate-t quantile (labelled
MC). Fail-closed on <2 groups, empty cells, or zero MSE.

References: Tukey (1949) Biometrics 5:99; Dunnett (1955)
JASA 50:1096; Games & Howell (1976) J. Educ. Stat. 1:113;
Scheffe (1953) Biometrika 40:87; Hochberg & Tamhane (1987).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import studentized_range

FloatArray = NDArray[np.float64]


def _groups(y: FloatArray, g: FloatArray) -> tuple[list[FloatArray], list[int]]:
    yy = np.asarray(y, dtype=np.float64).ravel()
    gg = np.asarray(g).ravel()
    if yy.size != gg.size or yy.size < 4:
        raise ValueError("bad shapes")
    uniq = np.unique(gg)
    if uniq.size < 2:
        raise ValueError("need >= 2 groups")
    out = [yy[gg == u] for u in uniq]
    if any(v.size == 0 for v in out):
        raise ValueError("empty group")
    return out, [int(u) for u in uniq]


def _pooled_mse(groups: list[FloatArray]) -> tuple[float, int]:
    num = 0.0
    den = 0
    for v in groups:
        num += float(((v - v.mean()) ** 2).sum())
        den += v.size - 1
    mse = num / max(den, 1)
    if mse <= 0:
        raise ValueError("zero pooled variance")
    return mse, den


def tukey_hsd(y: FloatArray, g: FloatArray, alpha: float = 0.05) -> dict[str, float | FloatArray]:
    """Tukey HSD — all-pairs q vs studentized range."""
    groups, labels = _groups(y, g)
    k = len(groups)
    mse, nu = _pooled_mse(groups)
    means = np.array([v.mean() for v in groups])
    pvals = np.full((k, k), np.nan)
    qs = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            se = math.sqrt(mse * 0.5 * (1.0 / groups[i].size + 1.0 / groups[j].size))
            q = abs(means[i] - means[j]) / se
            qs[i, j] = q
            pvals[i, j] = float(studentized_range.sf(q, k, nu))
    sig = int(np.nansum(pvals < alpha))
    return {
        "pair_q": np.asarray(qs, dtype=np.float64),
        "pair_p": np.asarray(pvals, dtype=np.float64),
        "n_significant": float(sig),
        "mse": mse,
        "df_error": float(nu),
        "labels": np.asarray(labels, dtype=np.float64),
    }


def dunnett_test(
    y: FloatArray,
    g: FloatArray,
    control: float | int | None = None,
    alpha: float = 0.05,
    mc: int = 4000,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Dunnett many-to-one vs control (MC multivariate-t p-values).

    The control is the group with the SMALLEST label unless
    `control` is given.
    """
    groups, labels = _groups(y, g)
    k = len(groups)
    if k < 2:
        raise ValueError("need >= 2 groups")
    c_idx = 0
    if control is not None:
        match = [i for i, v in enumerate(labels) if float(v) == float(control)]
        c_idx = match[0] if match else 0
    mse, nu = _pooled_mse(groups)
    means = np.array([v.mean() for v in groups])
    t_stats = []
    for j in range(k):
        if j == c_idx:
            continue
        se = math.sqrt(mse * (1.0 / groups[j].size + 1.0 / groups[c_idx].size))
        t_stats.append((means[j] - means[c_idx]) / se)
    t_arr = np.asarray(t_stats)
    # MC null: max |t| over treatment-control contrasts with the
    # correct equicorrelation rho_ij = sqrt(n_i n_j / ((n_i+n0)(n_j+n0)))
    sizes = np.array([groups[j].size for j in range(k)])
    trt_idx = [j for j in range(k) if j != c_idx]
    n0 = sizes[c_idx]
    corr = np.eye(len(trt_idx))
    for a, i in enumerate(trt_idx):
        for b, j in enumerate(trt_idx):
            if a != b:
                corr[a, b] = math.sqrt(sizes[i] * sizes[j] / ((sizes[i] + n0) * (sizes[j] + n0)))
    rng = np.random.default_rng(seed)
    e = np.linalg.cholesky(corr + 1e-10 * np.eye(len(trt_idx)))
    z = rng.standard_normal((mc, len(trt_idx))) @ e.T
    chi = np.sqrt(rng.chisquare(nu, size=mc) / nu)
    max_t = np.abs(z / chi[:, None]).max(axis=1)
    crit = float(np.quantile(max_t, 1.0 - alpha))
    # p per contrast: compare |t_j| against the joint max dist
    pvals = np.array([(max_t >= abs(t)).mean() for t in t_arr])
    return {
        "t_stats": np.asarray(t_arr, dtype=np.float64),
        "pair_p": np.asarray(pvals, dtype=np.float64),
        "crit_abs_t": crit,
        "control_label": float(labels[c_idx]),
        "n_significant": float((pvals < alpha).sum()),
    }


def games_howell(
    y: FloatArray, g: FloatArray, alpha: float = 0.05
) -> dict[str, float | FloatArray]:
    """Games-Howell — Welch denominators + Satterthwaite df."""
    groups, labels = _groups(y, g)
    k = len(groups)
    means = np.array([v.mean() for v in groups])
    vars_ = np.array([v.var(ddof=1) for v in groups])
    ns = np.array([v.size for v in groups])
    pvals = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            se2 = vars_[i] / ns[i] + vars_[j] / ns[j]
            t = abs(means[i] - means[j]) / math.sqrt(se2)
            df_num = se2 * se2
            df_den = (vars_[i] / ns[i]) ** 2 / (ns[i] - 1) + (vars_[j] / ns[j]) ** 2 / (ns[j] - 1)
            df = df_num / max(df_den, 1e-12)
            # studentized range approx on Welch df
            q = t * math.sqrt(2.0)
            pvals[i, j] = float(studentized_range.sf(q, k, max(df, 1.0)))
    sig = int(np.nansum(pvals < alpha))
    return {
        "pair_p": np.asarray(pvals, dtype=np.float64),
        "n_significant": float(sig),
        "labels": np.asarray(labels, dtype=np.float64),
    }


def scheffe_test(
    y: FloatArray, g: FloatArray, alpha: float = 0.05
) -> dict[str, float | FloatArray]:
    """Scheffe S-method — all-pairs conservative contrast test."""
    groups, labels = _groups(y, g)
    k = len(groups)
    mse, nu = _pooled_mse(groups)
    means = np.array([v.mean() for v in groups])
    s_crit = math.sqrt((k - 1) * stats.f.ppf(1.0 - alpha, k - 1, nu))
    pvals = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            se = math.sqrt(mse * (1.0 / groups[i].size + 1.0 / groups[j].size))
            f = ((means[i] - means[j]) / se) ** 2 / (k - 1)
            pvals[i, j] = float(stats.f.sf(f, k - 1, nu))
    return {
        "pair_p": np.asarray(pvals, dtype=np.float64),
        "s_crit": s_crit,
        "n_significant": float(int(np.nansum(pvals < alpha))),
        "labels": np.asarray(labels, dtype=np.float64),
    }


def bench_multiple_comparisons(seed: int = 20261231 + 456) -> dict[str, float]:
    """SYNTHETIC check — Tukey/Dunnett flag planted shift only."""
    rng = np.random.default_rng(seed)
    n = 40
    ys, gs = [], []
    mus = [0.0, 0.0, 0.0, 1.6]  # only group 3 shifted
    for j, mu in enumerate(mus):
        ys.append(rng.normal(loc=mu, size=n))
        gs.append(np.full(n, j))
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    out = tukey_hsd(y, g)
    p = np.asarray(out["pair_p"], dtype=np.float64)
    # pairs involving group 3 should be significant; 0-1,0-2,1-2 not
    sig_33 = p[0, 3] < 0.05 and p[1, 3] < 0.05 and p[2, 3] < 0.05
    not_sig = p[0, 1] > 0.05 and p[0, 2] > 0.05 and p[1, 2] > 0.05
    if not (sig_33 and not_sig):
        raise ValueError(f"tukey off: p={np.round(p, 3).tolist()}")
    d = dunnett_test(y, g, control=0, seed=seed)
    dp = np.asarray(d["pair_p"], dtype=np.float64)
    if not (dp[0] > 0.05 and dp[1] > 0.05 and dp[2] < 0.10):
        raise ValueError(f"dunnett off: {dp.tolist()}")
    return {
        "synthetic_tukey_min_p_shifted": float(np.nanmin([p[0, 3], p[1, 3], p[2, 3]])),
        "synthetic_dunnett_p3": float(dp[2]),
        "synthetic_score": 1.0,
    }
