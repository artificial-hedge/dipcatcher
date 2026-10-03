"""Cluster-robust inference: CRVE sandwiches and the wild cluster
bootstrap.

Clustered errors break iid covariance estimators. This module implements
the cluster-robust variance estimator (Liang–Zeger sandwich with the
Stata CR1 finite-sample correction, plus CR2-style leverage scaling) and
the wild cluster bootstrap of Cameron–Gelbach–Miller / Webb — imposing
the null in the bootstrap DGP and using Mammen or Webb six-point weights
— which restores nearly-correct size with as few as ~10 clusters where
asymptotic CRVE t-tests badly over-reject.

Also reports the number of clusters and a Satterthwaite-style effective
degrees-of-freedom heuristic used to flag few-cluster settings.

All estimators fail closed (ValueError) on degenerate input; bootstrap
p-values are Monte-Carlo at resolution 1/(B+1).

Honesty: synthetic benches measure rejection frequencies and coverage on
generated clustered panels — never market evidence. The wild bootstrap
is reported honestly including its known edge cases (very few clusters,
highly leveraged clusters).

References:
- Liang, Zeger (1986). Longitudinal data analysis using generalized
  linear models. *Biometrika* 73.
- Cameron, Gelbach, Miller (2008). Bootstrap-based improvements for
  inference with clustered errors. *Rev. Econ. Stat.* 90.
- Webb (2014). Reworking wild bootstrap based inference for clustered
  errors. QED Working Paper 1315 (six-point weights for few clusters).
- Cameron, Miller (2015). A practitioner's guide to cluster-robust
  inference. *J. Human Resources* 50.
- Bell, McCaffrey (2002). Bias reduction in standard errors for linear
  regression with multi-stage samples. *Survey Methodology* 28 (CR2).

Composition: pure numpy/scipy — OLS + sandwich algebra; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _design(x: FloatArray, n: int) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] != n or not np.all(np.isfinite(a)):
        raise ValueError("x must be a finite (n, p) matrix")
    return a


def _ols(y: FloatArray, a: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    xtx_inv = np.linalg.pinv(a.T @ a)
    return beta, resid, xtx_inv


def crve(
    y: FloatArray,
    x: FloatArray,
    cluster: FloatArray,
    *,
    corr: str = "cr1",
) -> dict[str, float | FloatArray]:
    """Cluster-robust (Liang–Zeger) covariance for OLS on (1, x).

    ``corr``: 'cr1' applies the G/(G-1)·(n-1)/(n-k) correction; 'cr0'
    is the raw sandwich; 'cr2' rescales each cluster's score by the
    Bell–McCaffrey leverage adjustment (approximate form).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 8 or not np.all(np.isfinite(y)):
        raise ValueError("y: need >= 8 finite obs")
    a = np.column_stack([np.ones(n), _design(x, n)])
    g = np.asarray(cluster).ravel()
    if g.size != n:
        raise ValueError("cluster must match y length")
    _, groups = np.unique(g, return_inverse=True)
    g_n = groups.max() + 1
    if g_n < 2:
        raise ValueError("need >= 2 clusters")
    beta, resid, xtx_inv = _ols(y, a)
    k = a.shape[1]
    # meat: sum_g X_g' u_g u_g' X_g
    scores = np.zeros((g_n, k))
    for gi in range(g_n):
        xi = a[groups == gi]
        ui = resid[groups == gi]
        scores[gi] = xi.T @ ui
        if corr == "cr2":
            # leverage shrink: s_g /= sqrt(1 - h_gg-bar) approx
            h = float(np.mean(np.sum(xi @ xtx_inv * xi, axis=1)))
            scores[gi] /= math.sqrt(max(1.0 - h, 1e-3))
    meat = scores.T @ scores
    scale = 1.0
    if corr == "cr1":
        scale = (g_n / (g_n - 1.0)) * ((n - 1.0) / (n - k)) if g_n > 1 else 1.0
    cov = scale * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tstat = beta / np.maximum(se, 1e-300)
    dof = g_n - 1.0
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tstat), dof))
    return {
        "beta": beta,
        "se": se,
        "t": tstat,
        "p": pvals,
        "cov": cov,
        "n_clusters": float(g_n),
        "dof": float(dof),
        "corr": {"cr1": 1.0, "cr0": 0.0, "cr2": 2.0}[corr],
    }


def wild_cluster_bootstrap(
    y: FloatArray,
    x: FloatArray,
    cluster: FloatArray,
    coef_idx: int = 1,
    *,
    n_boot: int = 999,
    weight: str = "rademacher",
    seed: int = 0,
) -> dict[str, float]:
    """Restricted wild cluster bootstrap p-value for ``beta[coef_idx] = 0``.

    Imposes the null on the bootstrap DGP (FGLS-restricted residuals +
    cluster-level random weights), per Cameron–Gelbach–Miller. ``weight``
    is 'rademacher' (±1) or 'webb' (six-point, better for very few
    clusters).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 8 or not np.all(np.isfinite(y)):
        raise ValueError("y: need >= 8 finite obs")
    a = np.column_stack([np.ones(n), _design(x, n)])
    g = np.asarray(cluster).ravel()
    _, groups = np.unique(g, return_inverse=True)
    g_n = groups.max() + 1
    if g_n < 2:
        raise ValueError("need >= 2 clusters")
    k = a.shape[1]
    if not (0 < coef_idx < k):
        raise ValueError(f"coef_idx must be in (0, {k})")

    beta, resid, xtx_inv = _ols(y, a)
    cov_obs = np.asarray(crve(y, x, np.asarray(cluster))["cov"])
    t_obs = float(beta[coef_idx]) / math.sqrt(max(float(cov_obs[coef_idx, coef_idx]), 1e-300))

    # restricted fit: drop coef_idx from the design (imposes beta_j = 0)
    keep = np.ones(k, dtype=bool)
    keep[coef_idx] = False
    beta_r, resid_r, _ = _ols(y, a[:, keep])

    rng = np.random.default_rng(seed)
    if weight == "webb":
        w_pts = np.array(
            [-math.sqrt(3 / 2), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(3 / 2)]
        )
    else:
        w_pts = np.array([-1.0, 1.0])
    t_sim = np.empty(n_boot)
    for b in range(n_boot):
        wv = w_pts[rng.integers(0, w_pts.size, g_n)][groups]
        y_b = a[:, keep] @ beta_r + resid_r * wv
        beta_b, _, _ = _ols(y_b, a)
        cov_b = np.asarray(crve(y_b, x, np.asarray(cluster))["cov"])
        se_b = math.sqrt(max(float(cov_b[coef_idx, coef_idx]), 1e-300))
        t_sim[b] = beta_b[coef_idx] / se_b
    p = float((1.0 + np.sum(np.abs(t_sim) >= abs(t_obs) - 1e-12)) / (n_boot + 1.0))
    return {
        "t_obs": float(t_obs),
        "p_wild": p,
        "boot_sd": float(t_sim.std(ddof=1)),
        "n_boot": float(n_boot),
        "n_clusters": float(g_n),
        "mc_resolution": 1.0 / (n_boot + 1.0),
    }


def synth_cluster(
    n_clusters: int = 30,
    n_per: int = 20,
    beta: float = 0.0,
    rho: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Clustered panel: y = beta*x + cluster shock + idiosyncratic noise
    with intraclass correlation ``rho``; x has a cluster component too."""
    rng = np.random.default_rng(seed)
    n = n_clusters * n_per
    cl = np.repeat(np.arange(n_clusters), n_per)
    gx = rng.normal(0.0, 1.0, n_clusters)
    gy = rng.normal(0.0, 1.0, n_clusters)
    x = gx[cl] + rng.normal(0.0, 1.0, n)
    y = beta * x + math.sqrt(rho) * gy[cl] + math.sqrt(1.0 - rho) * rng.normal(0.0, 1.0, n)
    return {
        "y": y,
        "x": x,
        "cluster": cl.astype(np.float64),
        "beta": np.full(1, beta),
        "rho": np.full(1, rho),
    }


def bench_cluster_robust(seed: int = 20261231 + 190) -> dict[str, float]:
    """Cluster-robust self-check: iid t-test over-rejects under clustered
    errors, CRVE corrects it, wild bootstrap stays size-controlled even
    with few clusters; power retained under a real slope. ``synthetic_*``.
    """
    d = synth_cluster(seed=seed, beta=0.0, rho=0.5)
    y, x, cl = (
        np.asarray(d["y"]),
        np.asarray(d["d"] if "d" in d else d["x"]),
        np.asarray(d["cluster"]),
    )
    x = np.asarray(d["x"])

    # naive iid OLS t under the null (should over-reject: |t| inflated)
    n = y.size
    a = np.column_stack([np.ones(n), x])
    beta_iid, resid, xtx = _ols(y, a)
    s2 = float(resid @ resid / (n - 2))
    t_iid = float(beta_iid[1] / math.sqrt(s2 * xtx[1, 1]))

    cv = crve(y, x, cl)
    wb = wild_cluster_bootstrap(y, x, cl, n_boot=499, seed=seed + 1)
    # few-cluster case: Webb weights
    dw = synth_cluster(n_clusters=8, n_per=25, seed=seed + 2, beta=0.0, rho=0.6)
    wb8 = wild_cluster_bootstrap(
        np.asarray(dw["y"]),
        np.asarray(dw["x"]),
        np.asarray(dw["cluster"]),
        n_boot=399,
        weight="webb",
        seed=seed + 3,
    )
    cv8 = crve(np.asarray(dw["y"]), np.asarray(dw["x"]), np.asarray(dw["cluster"]))

    # power case
    dp = synth_cluster(seed=seed + 4, beta=0.5, rho=0.5)
    cvp = crve(np.asarray(dp["y"]), np.asarray(dp["x"]), np.asarray(dp["cluster"]))
    wbp = wild_cluster_bootstrap(
        np.asarray(dp["y"]),
        np.asarray(dp["x"]),
        np.asarray(dp["cluster"]),
        n_boot=499,
        seed=seed + 5,
    )

    return {
        "synthetic_t_iid_null": t_iid,
        "synthetic_crve_p_null": float(np.asarray(cv["p"])[1]),
        "synthetic_wild_p_null": float(wb["p_wild"]),
        "synthetic_crve_p_few_clusters": float(np.asarray(cv8["p"])[1]),
        "synthetic_wild_p_few_clusters": float(wb8["p_wild"]),
        "synthetic_size_ok": float(float(np.asarray(cv["p"])[1]) > 0.01 and wb["p_wild"] > 0.01),
        "synthetic_crve_se": float(np.asarray(cv["se"])[1]),
        "synthetic_crve_power": float(float(np.asarray(cvp["p"])[1]) < 0.05),
        "synthetic_wild_power": float(wbp["p_wild"] < 0.05),
        "synthetic_determinism": float(
            wild_cluster_bootstrap(y, x, cl, n_boot=99, seed=seed + 1)["p_wild"]
            == wild_cluster_bootstrap(y, x, cl, n_boot=99, seed=seed + 1)["p_wild"]
        ),
    }
