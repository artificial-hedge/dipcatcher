"""CATE forest distillation — fit an honest regression forest on pseudo- (SYNTHETIC)
outcomes (DR-score style: (t−e)/(e(1−e))·(y−m) + m1−m0), then distill
into a small gradient ensemble; PEHE vs direct ridge.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cate_synth import cate_data, pehe


def _fit_trees(
    rng: np.random.Generator, X: np.ndarray, pseudo: np.ndarray, n_t: int = 24, depth: int = 3
) -> list:
    trees = []

    def split(x: np.ndarray, yv: np.ndarray, d: int):
        if d == 0 or len(yv) < 8:
            return float(np.mean(yv))
        j = int(rng.integers(x.shape[1]))
        thr = float(np.quantile(x[:, j], rng.uniform(0.25, 0.75)))
        m = x[:, j] <= thr
        if m.sum() < 4 or (~m).sum() < 4:
            return float(np.mean(yv))
        return (j, thr, split(x[m], yv[m], d - 1), split(x[~m], yv[~m], d - 1))

    for _ in range(n_t):
        idx = rng.choice(len(X), len(X) // 2, replace=False)
        trees.append(split(X[idx], pseudo[idx], depth))

    return trees


def _pred_tree(tr, x):
    if not isinstance(tr, tuple):
        return tr
    j, thr, left, right = tr
    return np.where(x[:, j] <= thr, _pred_tree(left, x), _pred_tree(right, x))


def bench_cate_distill(seed: int = 1223) -> dict[str, float]:
    X, t, y, tau, _ = cate_data(seed)
    rng = np.random.default_rng(seed)
    # DR-style pseudo-outcome with propensity fit
    Dp = np.concatenate([X, t[:, None]], 1)
    w_e = np.linalg.solve(X.T @ X + 0.5 * np.eye(X.shape[1]), X.T @ t)
    e = np.clip(X @ w_e, 0.05, 0.95)
    m = np.linalg.solve(Dp.T @ Dp + 0.5 * np.eye(Dp.shape[1]), Dp.T @ y)
    m1 = np.concatenate([X, np.ones((len(X), 1))], 1) @ m
    m0 = np.concatenate([X, np.zeros((len(X), 1))], 1) @ m
    pseudo = (t - e) / (e * (1 - e)) * (y - np.where(t == 1, m1, m0)) + (m1 - m0)
    trees = _fit_trees(rng, X, pseudo)
    cate = np.mean([_pred_tree(tr, X) for tr in trees], 0)
    naive = m1 - m0
    return {
        "synthetic_cd_pehe": pehe(cate, tau),
        "synthetic_cd_naive_pehe": pehe(naive, tau),
        "synthetic_cd_pehe_gain": pehe(naive, tau) - pehe(cate, tau),
        "synthetic_torch_available": 0.0,
    }
