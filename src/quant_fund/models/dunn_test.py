"""Dunn's post-hoc test — pairwise rank comparisons after Kruskal.

Dunn (1964): after a significant Kruskal-Wallis omnibus, pairwise
group comparisons on mean-rank differences:

    z_ij = (Rbar_i - Rbar_j) /
           sqrt( [N(N+1)/12 - tie-corr] (1/n_i + 1/n_j) )

with tie correction sum_g (t_g^3 - t_g) / (12 (N-1)); two-sided
normal p's adjusted across the k(k-1)/2 family by Bonferroni or
Holm (1979) step-down. Honest post-hoc — do not run when the
omnibus is non-significant without noting the family error.

Honesty: the bench plants two shifted groups among three (the
shifted-vs-null pairs must reject after Holm; the null-null pair
must not). Fail-closed on <2 groups, tiny groups, or all-tied
pools.

References: Dunn (1964) "Multiple comparisons using rank sums",
Technometrics 6:241; Holm (1979) "A simple sequentially rejective
multiple test procedure", Scand. J. Stat. 6:65; Conover (1999)
ch. 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

FloatArray = NDArray[np.float64]


def _groups(groups: tuple[FloatArray, ...]) -> list[FloatArray]:
    gs = [np.asarray(g, dtype=float) for g in groups]
    if len(gs) < 2:
        raise ValueError("need >=2 groups")
    for g in gs:
        if g.ndim != 1 or g.size < 5 or not np.isfinite(g).all():
            raise ValueError("bad group")
    return gs


def _holm_adjust(ps: FloatArray) -> FloatArray:
    """Holm step-down adjusted p-values (order-free)."""
    m = ps.size
    order = np.argsort(ps, kind="stable")
    adj = np.empty(m)
    run_max = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * float(ps[idx])
        run_max = max(run_max, min(1.0, val))
        adj[idx] = run_max
    return np.minimum(adj, 1.0)


def dunn_test(*groups: FloatArray, adjust: str = "holm") -> dict[str, float | FloatArray]:
    """Pairwise Dunn z-tests among >=2 groups.

    Returns parallel arrays ``pairs`` (i,j), ``z``, ``p_raw``, and
    ``p_adj`` (Holm step-down, or Bonferroni with
    ``adjust='bonferroni'``), plus the family size.
    """
    gs = _groups(groups)
    k = len(gs)
    pooled = np.concatenate(gs)
    n = pooled.size
    r = rankdata(pooled)
    idx = np.concatenate([[i] * g.size for i, g in enumerate(gs)])
    rbar = np.array([r[idx == i].mean() for i in range(k)])
    # tie correction
    _, counts = np.unique(pooled, return_counts=True)
    tie = float((counts**3 - counts).sum())
    sigma_sq = n * (n + 1) / 12.0 - tie / (12.0 * (n - 1))
    if sigma_sq <= 0:
        raise ValueError("all tied")
    pairs: list[tuple[int, int]] = []
    zs: list[float] = []
    ps: list[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            se = np.sqrt(sigma_sq * (1.0 / gs[i].size + 1.0 / gs[j].size))
            z = (rbar[i] - rbar[j]) / se
            pairs.append((i, j))
            zs.append(float(z))
            ps.append(float(2.0 * norm.sf(abs(z))))
    p_raw = np.asarray(ps)
    if adjust == "holm":
        p_adj = _holm_adjust(p_raw)
    elif adjust == "bonferroni":
        p_adj = np.minimum(p_raw * len(ps), 1.0)
    else:
        raise ValueError("unknown adjust")
    return {
        "n_pairs": float(len(pairs)),
        "pairs": np.asarray(pairs, dtype=np.float64),
        "z": np.asarray(zs, dtype=np.float64),
        "p_raw": p_raw,
        "p_adj": np.asarray(p_adj, dtype=np.float64),
    }


def bench_dunn_test(seed: int = 20261231 + 437) -> dict[str, float]:
    """SYNTHETIC check — shifted pairs rejected, null pair held."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(60)
    b = rng.standard_normal(60) + 1.3
    c = rng.standard_normal(60) + 1.3
    out = dunn_test(a, b, c)
    pairs = np.asarray(out["pairs"], dtype=int)
    p_adj = np.asarray(out["p_adj"], dtype=float)
    # pair (0,1) and (0,2) must reject; (1,2) must not
    p_ab = float(p_adj[np.where((pairs == [0, 1]).all(axis=1))[0][0]])
    p_ac = float(p_adj[np.where((pairs == [0, 2]).all(axis=1))[0][0]])
    p_bc = float(p_adj[np.where((pairs == [1, 2]).all(axis=1))[0][0]])
    if p_ab > 0.05 or p_ac > 0.05 or p_bc < 0.05:
        raise ValueError(f"dunn off: ab={p_ab:.4f} ac={p_ac:.4f} bc={p_bc:.4f}")
    return {
        "synthetic_dunn_p_ab": p_ab,
        "synthetic_dunn_p_ac": p_ac,
        "synthetic_dunn_p_bc": p_bc,
        "synthetic_score": 1.0,
    }
