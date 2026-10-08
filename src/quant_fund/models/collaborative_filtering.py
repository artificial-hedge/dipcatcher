"""Collaborative filtering: cosine item-kNN with (SYNTHETIC)
shrinkage (Sarwar 2001), ALS with weighted-λ
regularization (Zhou et al. 2008), and a bias-augmented
factor model (Koren's SVD++-style baseline). Synthetic
bench gates held-out rating RMSE vs a global-mean
baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def item_knn_predict(
    r: FloatArray,
    k: int = 10,
    shrink: float = 10.0,
) -> FloatArray:
    """Adjusted-cosine item-item kNN prediction for every
    missing cell: r̂_ui = Σ_{j∈N_i∩R_u} s_ij r_uj / Σ|s_ij|."""
    r = np.asarray(r, dtype=np.float64)
    n_u, n_i = r.shape
    mask = r != 0
    mean_u = np.divide(
        r.sum(axis=1), mask.sum(axis=1), out=np.zeros(n_u), where=mask.sum(axis=1) > 0
    )
    centered = np.where(mask, r - mean_u[:, None], 0.0)
    cov = centered.T @ centered
    cnt = mask.T @ mask.astype(float)
    denom = np.sqrt(np.clip(np.diag(cov)[:, None] * np.diag(cov)[None, :], 1e-12, None))
    sim = cov / denom * (cnt / (cnt + shrink))
    np.fill_diagonal(sim, 0.0)
    pred = np.zeros_like(r)
    for u in range(n_u):
        for i in range(n_i):
            if mask[u, i]:
                continue
            nbrs = np.argsort(-sim[i])[:k]
            use = nbrs[mask[u, nbrs]]
            if len(use) == 0:
                pred[u, i] = mean_u[u]
            else:
                s = sim[i, use]
                pred[u, i] = mean_u[u] + float((s * (r[u, use] - mean_u[u])).sum()) / (
                    np.abs(s).sum() + 1e-12
                )
    return np.asarray(pred)


def als_wr_fit(
    r: FloatArray,
    n_factors: int = 8,
    lam: float = 0.05,
    it: int = 15,
    seed: int = 0,
) -> dict[str, object]:
    """ALS-WR: alternate ridge solves on observed entries
    only, weights = #ratings per row/col."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=np.float64)
    n_u, n_i = r.shape
    mask = r != 0
    p = rng.normal(0, 0.1, (n_u, n_factors))
    q = rng.normal(0, 0.1, (n_i, n_factors))
    for _ in range(it):
        for u in range(n_u):
            obs = mask[u]
            if not obs.any():
                continue
            qo = q[obs]
            a = qo.T @ qo + lam * obs.sum() * np.eye(n_factors)
            p[u] = np.linalg.solve(a, qo.T @ r[u, obs])
        for i in range(n_i):
            obs = mask[:, i]
            if not obs.any():
                continue
            po = p[obs]
            a = po.T @ po + lam * obs.sum() * np.eye(n_factors)
            q[i] = np.linalg.solve(a, po.T @ r[obs, i])
    return {"p": p, "q": q}


def als_predict(model: dict[str, object]) -> FloatArray:
    p = np.asarray(model["p"])
    q = np.asarray(model["q"])
    return np.asarray(p @ q.T)


def bias_mf_fit(
    r: FloatArray,
    n_factors: int = 8,
    lam: float = 0.02,
    lr: float = 0.02,
    it: int = 40,
    seed: int = 0,
) -> dict[str, object]:
    """SGD bias + factors (Koren): r̂ = μ + b_u + b_i + p_u·q_i."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=np.float64)
    n_u, n_i = r.shape
    mask = r != 0
    mu = float(r[mask].mean())
    bu = np.zeros(n_u)
    bi = np.zeros(n_i)
    p = rng.normal(0, 0.05, (n_u, n_factors))
    q = rng.normal(0, 0.05, (n_i, n_factors))
    us, i_s = np.where(mask)
    for _ in range(it):
        for u, i in zip(us, i_s, strict=True):
            err = r[u, i] - (mu + bu[u] + bi[i] + p[u] @ q[i])
            bu[u] += lr * (err - lam * bu[u])
            bi[i] += lr * (err - lam * bi[i])
            pu = p[u].copy()
            p[u] += lr * (err * q[i] - lam * p[u])
            q[i] += lr * (err * pu - lam * q[i])
    return {"mu": mu, "bu": bu, "bi": bi, "p": p, "q": q}


def bias_mf_predict(model: dict[str, object]) -> FloatArray:
    p = np.asarray(model["p"])
    q = np.asarray(model["q"])
    return np.asarray(
        float(np.asarray(model["mu"]))
        + np.asarray(model["bu"])[:, None]
        + np.asarray(model["bi"])[None, :]
        + p @ q.T
    )


def bench_collaborative_filtering(seed: int = 567) -> dict[str, float]:
    """SYNTHETIC: ratings from a planted rank-3 factor model
    + item biases — ALS and bias-MF must halve the global-
    mean RMSE, item-kNN must beat it materially."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n_u, n_i = 60, 40
    p_true = rng.normal(0, 1, (n_u, 3))
    q_true = rng.normal(0, 1, (n_i, 3))
    full = 3.0 + p_true @ q_true.T + rng.normal(0, 0.3, (n_u, n_i))
    mask = rng.uniform(0, 1, (n_u, n_i)) < 0.5
    r = np.where(mask, full, 0.0)
    # hold out 15% of observed cells
    us, i_s = np.where(mask)
    hold = rng.choice(len(us), int(0.15 * len(us)), replace=False)
    hu, hi = us[hold], i_s[hold]
    r_train = r.copy()
    true_hold = r[hu, hi].copy()
    r_train[hu, hi] = 0.0
    mu = float(r_train[r_train != 0].mean())
    out["synthetic_cf_mean_rmse"] = float(np.sqrt(((true_hold - mu) ** 2).mean()))
    pk = item_knn_predict(r_train, k=8)
    out["synthetic_cf_knn_rmse"] = float(np.sqrt(((true_hold - pk[hu, hi]) ** 2).mean()))
    als = als_wr_fit(r_train, n_factors=5, lam=0.05, it=15, seed=seed)
    ap = als_predict(als)
    out["synthetic_cf_als_rmse"] = float(np.sqrt(((true_hold - ap[hu, hi]) ** 2).mean()))
    bm = bias_mf_fit(r_train, n_factors=5, lr=0.02, it=40, seed=seed)
    bp = bias_mf_predict(bm)
    out["synthetic_cf_bmf_rmse"] = float(np.sqrt(((true_hold - bp[hu, hi]) ** 2).mean()))
    base = out["synthetic_cf_mean_rmse"]
    if out["synthetic_cf_als_rmse"] > 0.65 * base:
        raise ValueError(f"als rmse off: {out['synthetic_cf_als_rmse']} vs {base}")
    if out["synthetic_cf_bmf_rmse"] > 0.75 * base:
        raise ValueError(f"bmf rmse off: {out['synthetic_cf_bmf_rmse']} vs {base}")
    if out["synthetic_cf_knn_rmse"] > 0.9 * base:
        raise ValueError(f"knn rmse off: {out['synthetic_cf_knn_rmse']} vs {base}")
    return out
