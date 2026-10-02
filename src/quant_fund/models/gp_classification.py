"""Binary Gaussian-process classification under the Laplace
approximation (Rasmussen & Williams ch. 3): Newton mode-finding
on the latent logit, Gaussian predictive via the
(K + W⁻¹)⁻¹ form. Synthetic bench gates two-moons AUC vs a
linear-logit baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _sigmoid(z: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def _rbf(x: FloatArray, y: FloatArray, ls: float, sf2: float) -> FloatArray:
    d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(sf2 * np.exp(-d2 / (2 * ls * ls)))


def gpc_fit(
    x: FloatArray,
    y: FloatArray,
    ls: float = 1.0,
    sf2: float = 2.0,
    it: int = 60,
    tol: float = 1e-6,
) -> dict[str, object]:
    """Newton iterations on posterior mode f̂ of the latent."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = _rbf(x, x, ls, sf2) + 1e-6 * np.eye(x.shape[0])
    f = np.zeros(x.shape[0])
    w = np.full(x.shape[0], 0.25)
    for _ in range(it):
        p = _sigmoid(f)
        w = np.maximum(p * (1 - p), 1e-4)
        # f_new = (K⁻¹+W)⁻¹ (∇log p + Wf) = K (I+WK)⁻¹ (y−p+Wf)
        b = w * f + y - p
        wb = w * b
        inner = np.linalg.solve(np.eye(x.shape[0]) + w[:, None] * k, wb)
        f_new = k @ inner
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    return {"x": x, "f": f, "w": w, "k": k, "ls": ls, "sf2": sf2}


def gpc_predict(model: dict[str, object], x_new: FloatArray) -> FloatArray:
    """Latent predictive mean → sigmoid probabilities."""
    x = np.asarray(model["x"])
    f = np.asarray(model["f"])
    k = np.asarray(model["k"])
    w = np.asarray(model["w"])
    ls = float(np.asarray(model["ls"]))
    sf2 = float(np.asarray(model["sf2"]))
    x_new = np.asarray(x_new, dtype=np.float64)
    ks = _rbf(x, x_new, ls, sf2)
    alpha = np.linalg.solve(k, f)
    f_star = ks.T @ alpha
    # predictive var via (K + W⁻¹)⁻¹
    winv = np.linalg.inv(np.diag(w) + 1e-8 * np.eye(len(w)))
    big = np.linalg.inv(k + winv)
    v = np.diag(_rbf(x_new, x_new, ls, sf2)) - np.einsum("ij,jk,ki->i", ks.T, big, ks)
    f_adj = f_star / np.sqrt(1.0 + np.maximum(v, 0) * np.pi / 8)
    return _sigmoid(np.asarray(f_adj))


def _auc(scores: FloatArray, labels: FloatArray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores)) + 1
    pos = labels.astype(bool)
    n_pos = int(pos.sum())
    if n_pos == 0 or n_pos == len(scores):
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (~pos).sum()))


def bench_gp_classification(seed: int = 549) -> dict[str, float]:
    """SYNTHETIC: two-moons — GPC AUC must beat a linear
    logistic baseline."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 120
    t = rng.uniform(0, np.pi, n)
    x = np.vstack(
        [
            np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.08, (n, 2)),
            np.c_[1 - np.cos(t), 0.5 - np.sin(t)] + rng.normal(0, 0.08, (n, 2)),
        ]
    )
    y = np.r_[np.zeros(n), np.ones(n)]
    perm = rng.permutation(2 * n)
    tr, te = perm[: 2 * n // 2], perm[2 * n // 2 :]
    mdl = gpc_fit(x[tr], y[tr], ls=0.6, sf2=2.0)
    p = gpc_predict(mdl, x[te])
    auc = _auc(np.asarray(p), y[te])
    out["synthetic_gpc_auc"] = auc
    # linear logit baseline (Newton, few iters)
    b = np.zeros(3)
    xa = np.c_[x[tr], np.ones(tr.shape[0])]
    for _ in range(50):
        q = _sigmoid(xa @ b)
        b += np.linalg.solve(
            xa.T @ (xa * (q * (1 - q))[:, None]) + 0.01 * np.eye(3), xa.T @ (y[tr] - q)
        )
    q_te = _sigmoid(np.c_[x[te], np.ones(te.shape[0])] @ b)
    auc_lin = _auc(np.asarray(q_te), y[te])
    out["synthetic_logit_auc"] = auc_lin
    if auc < 0.9:
        raise ValueError(f"gpc auc off: {auc}")
    if auc <= auc_lin:
        raise ValueError(f"gpc not > logit: {auc} vs {auc_lin}")
    return out
