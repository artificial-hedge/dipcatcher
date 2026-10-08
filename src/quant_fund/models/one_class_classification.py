"""One-class classification: SVDD (Tax & Duin 2004) with a (SYNTHETIC)
quadratic-program-lite center/radius fit, Mahalanobis
outlier scoring with shrinkage covariance, and local
outlier factor (Breunig et al. 2000). Synthetic bench gates
planted-outlier AUC."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def svdd_fit(
    x: FloatArray, nu: float = 0.1, it: int = 400, lr: float = 0.1, seed: int = 0
) -> dict[str, object]:
    """SVDD via subgradient descent on the soft-ball objective
    R² + (1/νn)Σ ξ_i, ξ_i = max(0, ‖x_i − c‖² − R²)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    c = x[rng.integers(n)].copy()
    r2 = float(np.percentile(((x - c) ** 2).sum(axis=1), 50))
    for t in range(it):
        step = lr / np.sqrt(t + 1)
        d2 = ((x - c) ** 2).sum(axis=1)
        inside = d2 <= r2
        c = c * inside.mean() + (x[inside].mean(axis=0) if inside.any() else c)
        # simpler: c = mean of points inside ball (SVDD KKT form)
        viol = ~inside
        r2 = r2 * (1 - step * (1 - viol.mean() / max(nu, 1e-3)))
        if viol.any():
            c += step * (x[viol].mean(axis=0) - c) * viol.mean()
    d2 = ((x - c) ** 2).sum(axis=1)
    r2 = float(np.quantile(d2, 1 - nu))
    return {"center": c, "r2": r2}


def svdd_score(model: dict[str, object], x: FloatArray) -> FloatArray:
    """Squared distance from the SVDD center minus R² —
    positive outside."""
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(model["center"])
    r2 = float(np.asarray(model["r2"]))
    return np.asarray(((x - c) ** 2).sum(axis=1) - r2)


def mahalanobis_score(x_tr: FloatArray, x_te: FloatArray, shrink: float = 0.1) -> FloatArray:
    """Squared Mahalanobis distance to the shrunken train
    covariance."""
    x_tr = np.asarray(x_tr, dtype=np.float64)
    x_te = np.asarray(x_te, dtype=np.float64)
    mu = x_tr.mean(axis=0)
    cov = np.cov(x_tr.T)
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    prec = np.linalg.inv(cov + 1e-8 * np.eye(cov.shape[0]))
    diff = x_te - mu
    return np.asarray(np.einsum("ij,jk,ik->i", diff, prec, diff))


def lof_score(x: FloatArray, k: int = 10) -> FloatArray:
    """LOF: local reachability density ratio — mean lrd of
    k-NN neighbors over own lrd."""
    x = np.asarray(x, dtype=np.float64)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    kdist = np.partition(d, k - 1, axis=1)[:, k - 1]
    nn = np.argsort(d, axis=1)[:, :k]
    # reachability distance reach(a,b) = max(kdist(b), d(a,b))
    rd = np.maximum(kdist[nn], np.take_along_axis(d, nn, axis=1))
    lrd = 1.0 / np.maximum(rd.mean(axis=1), 1e-12)
    lof = lrd[nn].mean(axis=1) / lrd
    return np.asarray(lof)


def _auc(scores: FloatArray, labels: FloatArray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores)) + 1
    pos = labels.astype(bool)
    n_pos = int(pos.sum())
    if n_pos == 0 or n_pos == len(scores):
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (~pos).sum()))


def bench_one_class(seed: int = 548) -> dict[str, float]:
    """SYNTHETIC: tight inlier blob + planted outliers —
    SVDD/Mahalanobis/LOF each must hit AUC > 0.85."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n_in, n_out = 200, 25
    x_in = rng.normal([0, 0], 0.5, (n_in, 2))
    x_out = rng.uniform(-4, 4, (n_out, 2))
    # keep outliers genuinely outlying
    x_out = x_out[np.linalg.norm(x_out, axis=1) > 2.5]
    n_out = len(x_out)
    x = np.vstack([x_in, x_out])
    y = np.r_[np.zeros(n_in), np.ones(n_out)]
    mdl = svdd_fit(x_in, nu=0.1, seed=seed)
    s_svdd = svdd_score(mdl, x)
    out["synthetic_svdd_auc"] = _auc(s_svdd, y)
    s_m = mahalanobis_score(x_in, x)
    out["synthetic_maha_auc"] = _auc(s_m, y)
    s_lof = lof_score(x, k=10)
    out["synthetic_lof_auc"] = _auc(s_lof, y)
    for k_, a in (
        ("svdd", out["synthetic_svdd_auc"]),
        ("maha", out["synthetic_maha_auc"]),
        ("lof", out["synthetic_lof_auc"]),
    ):
        if a < 0.85:
            raise ValueError(f"{k_} auc off: {a}")
    return out
