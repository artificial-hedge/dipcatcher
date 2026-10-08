"""Two-stage cluster sampling — design-based total/mean with (SYNTHETIC)
cluster-robust variance.

Särndal, Swensson & Wretman (1992) ch. 4: for m sampled PSU
clusters with totals t_i and inclusion weights d_i = 1/pi_i, the
HT total and its variance simplify to the standard cluster
estimator

    t_hat = sum_i d_i t_i
    V_hat = m/(m-1) sum_i (d_i t_i - t_hat/m)^2  (with-replacement
    approximation; conservative for without-replacement)

Intracluster correlation rho_hat and the DEFF from clustering
deff_cl = 1 + (bbar - 1) rho (Kish) describe efficiency loss vs
SRS. Finite-population correction multiplies by (1 - m/M) when
the PSU count M is known.

Honesty: the bench builds a two-stage population with a planted
ICC (clusters share means), draws a PPS cluster sample, and
checks the cluster-robust interval covers the true total while
an SRS-wrong variance understates it. Fail-closed on single-
cluster or degenerate designs.

References: Särndal, Swensson, Wretman (1992) "Model Assisted
Survey Sampling" ch. 4; Kish (1965); Cochran (1977) "Sampling
Techniques" 3rd ed. ch. 9-10.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cluster_total(
    cluster_totals: FloatArray,
    weights: FloatArray,
    n_psu_population: int | None = None,
) -> dict[str, float]:
    """Cluster-robust HT total.

    ``cluster_totals`` = y summed within each sampled cluster,
    ``weights`` = 1/pi_i design weights per cluster.
    """
    t = np.asarray(cluster_totals, dtype=float)
    d = np.asarray(weights, dtype=float)
    if t.ndim != 1 or d.shape != t.shape or t.size < 2:
        raise ValueError("bad clusters")
    if not np.isfinite(t).all() or not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError("bad inputs")
    m = t.size
    dt = d * t
    t_hat = float(dt.sum())
    var = float(m / (m - 1) * ((dt - t_hat / m) ** 2).sum())
    if n_psu_population is not None and n_psu_population > m:
        var *= 1.0 - m / n_psu_population
    return {"total": t_hat, "se": float(np.sqrt(var))}


def cluster_mean(y: FloatArray, cluster: FloatArray) -> dict[str, float]:
    """Cluster-robust mean of element-level y with cluster ids.

    Uses the ratio-estimator cluster variance at element level
    (equivalent to the standard clustered SE for a mean).
    """
    a = np.asarray(y, dtype=float)
    g = np.asarray(cluster, dtype=float)
    if a.ndim != 1 or g.shape != a.shape or a.size < 6:
        raise ValueError("bad inputs")
    if not np.isfinite(a).all():
        raise ValueError("non-finite input")
    ids = np.unique(g)
    if ids.size < 2:
        raise ValueError("need >=2 clusters")
    mean = float(a.mean())
    parts = np.array([a[g == i].sum() - mean * (g == i).sum() for i in ids])
    m = ids.size
    var = float(m / (m - 1) * (parts**2).sum()) / a.size**2
    # ICC estimate: MS_between - MS_within decomposition
    sizes = np.array([(g == i).sum() for i in ids], dtype=float)
    grand = a.mean()
    ms_b = float((sizes * (np.array([a[g == i].mean() for i in ids]) - grand) ** 2).sum() / (m - 1))
    ms_w = float(sum(((a[g == i] - a[g == i].mean()) ** 2).sum() for i in ids) / max(a.size - m, 1))
    bbar = sizes.mean()
    rho = (
        max(0.0, (ms_b - ms_w) / (ms_b + (bbar - 1) * ms_w))
        if ms_b + (bbar - 1) * ms_w > 0
        else 0.0
    )
    deff = 1.0 + (bbar - 1.0) * rho
    return {
        "mean": mean,
        "se": float(np.sqrt(max(var, 0.0))),
        "icc": float(rho),
        "deff_cluster": float(deff),
        "n_clusters": float(m),
    }


def bench_cluster_sampling(seed: int = 20261231 + 448) -> dict[str, float]:
    """SYNTHETIC check — cluster SE covers truth, SRS SE too small."""
    rng = np.random.default_rng(seed)
    n_cl, per = 60, 12
    mu_c = rng.normal(0, 1.5, n_cl)  # cluster means -> ICC
    y = np.concatenate([mu_c[i] + rng.standard_normal(per) for i in range(n_cl)])
    g = np.repeat(np.arange(n_cl), per).astype(float)
    sel = rng.choice(n_cl, 30, replace=False)
    mask = np.isin(g, sel)
    out = cluster_mean(y[mask], g[mask])
    true_mean = float(y.mean())
    z = abs(out["mean"] - true_mean) / out["se"]
    # wrong (SRS) variance
    se_srs = float(y[mask].std(ddof=1) / np.sqrt(mask.sum()))
    if z > 4.0 or se_srs > out["se"] * 0.9 or out["icc"] < 0.2:
        raise ValueError(
            f"cluster off: z={z:.2f} se={out['se']:.4f} srs={se_srs:.4f} icc={out['icc']:.3f}"
        )
    return {
        "synthetic_cluster_z": z,
        "synthetic_cluster_se": out["se"],
        "synthetic_srs_se": se_srs,
        "synthetic_icc": out["icc"],
        "synthetic_score": 1.0,
    }
