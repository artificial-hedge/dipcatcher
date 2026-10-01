"""Honest causal trees and a small generalized random forest.

Heterogeneous treatment-effect estimation via the *honest* protocol of
Athey & Imbens (2016): the sample is split in two — one half chooses
the tree structure, the other half estimates leaf effects — so leaf
effects aren't biased by adaptive splitting. The split criterion
maximizes the between-leaf variance of the estimated treatment effect
(the "TOT" criterion: transform-outcome / leaf-effect heterogeneity).

``causal_forest`` averages ``B`` honest trees grown on subsamples and
returns per-point CATE, a leaf-membership-based standard error, and a
global split-based variable-importance profile.

References
----------
- Athey, S. & Imbens, G. (2016). *Recursive partitioning for
  heterogeneous causal effects.* PNAS 113(27), 7353–7360.
- Wager, S. & Athey, S. (2018). *Estimation and inference of
  heterogeneous treatment effects using random forests.* JASA
  113(523), 1228–1242.
- Athey, S., Tibshirani, J. & Wager, S. (2019). *Generalized random
  forests.* Annals of Statistics 47(2), 1148–1178.

Honesty contract
----------------
``synth_*`` helpers and ``bench_*`` emit SYNTHETIC correctness checks
only — never market evidence.

Composition notes
-----------------
numpy/scipy only; deterministic ``np.random.default_rng(seed)``;
fail-closed ``ValueError`` on degenerate treatment variation or
samples too small to split honestly; trees are shallow
(``max_depth`` capped) — this is a workhorse, not a GBM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FloatArray = np.ndarray

__all__ = [
    "bench_causal_forest",
    "causal_forest",
    "causal_tree",
    "synth_cate",
]


@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    left: _Node | None = None
    right: _Node | None = None
    est_idx: FloatArray = field(default_factory=lambda: np.empty(0, dtype=int))
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def _check_inputs(
    y: FloatArray, t: FloatArray, x: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 1 or t.shape != y.shape or x.shape[0] != y.size:
        raise ValueError("y, t same-length; x same rows")
    if y.size < 80:
        raise ValueError("need at least 80 observations")
    if not {0.0, 1.0} <= set(np.unique(t.astype(int)).tolist()):
        raise ValueError("t must be binary")
    return y, t, x


def _transform_outcome(y: FloatArray, t: FloatArray, x: FloatArray) -> FloatArray:
    """Athey-Imbens transformed outcome ``z = y·t/e − y·(1−t)/(1−e)``
    — E[z|x] = CATE(x) under unconfoundedness, so leaf means of z are
    confounding-robust effect estimates. Propensity via ridge
    regression on 2-degree features, clipped."""
    f = np.column_stack(
        [np.ones(x.shape[0]), x]
        + [x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])]
    )
    b = np.linalg.lstsq(f.T @ f + 1e-3 * np.eye(f.shape[1]), f.T @ t, rcond=None)[0]
    e = np.clip(f @ b, 0.05, 0.95)
    return np.asarray(y * t / e - y * (1.0 - t) / (1.0 - e))


def _leaf_effect(z: FloatArray, _t: FloatArray, idx: FloatArray) -> float:
    """Leaf treatment-effect estimate = mean transformed outcome."""
    if idx.size < 3:
        return 0.0
    return float(z[idx].mean())


def _split_gain(
    z: FloatArray, t: FloatArray, idx: FloatArray, feat: int, thr: float, x: FloatArray
) -> float:
    """Between-leaf variance of the treatment effect (TOT proxy):
    gain = n·Var(tau_left, tau_right) weighted by leaf sizes — the
    standard maximizing-heterogeneity criterion."""
    lft = idx[x[idx, feat] <= thr]
    rgt = idx[x[idx, feat] > thr]
    if lft.size < 10 or rgt.size < 10:
        return -1.0
    tl = _leaf_effect(z, t, lft)
    tr = _leaf_effect(z, t, rgt)
    w_l = lft.size / idx.size
    w_r = rgt.size / idx.size
    mean = w_l * tl + w_r * tr
    return float(w_l * (tl - mean) ** 2 + w_r * (tr - mean) ** 2)


def _grow(
    z: FloatArray,
    t: FloatArray,
    x: FloatArray,
    split_idx: FloatArray,
    est_idx: FloatArray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    n_thr: int,
    rng: np.random.Generator,
) -> _Node:
    node = _Node(est_idx=est_idx, depth=depth)
    if depth >= max_depth or split_idx.size < 4 * min_leaf:
        return node
    best_feat, best_thr, best_gain = -1, 0.0, 0.0
    feats = rng.permutation(x.shape[1])[: max(1, int(math.sqrt(x.shape[1])) + 1)]
    for f in feats:
        col = x[split_idx, f]
        qs = np.quantile(col, np.linspace(0.1, 0.9, n_thr))
        for thr in np.unique(qs):
            g = _split_gain(z, t, split_idx, f, float(thr), x)
            if g > best_gain:
                best_gain, best_feat, best_thr = g, f, float(thr)
    if best_feat < 0 or best_gain <= 1e-10:
        return node
    node.feature = best_feat
    node.threshold = best_thr
    s_l = split_idx[x[split_idx, best_feat] <= best_thr]
    s_r = split_idx[x[split_idx, best_feat] > best_thr]
    e_l = est_idx[x[est_idx, best_feat] <= best_thr]
    e_r = est_idx[x[est_idx, best_feat] > best_thr]
    if e_l.size < min_leaf or e_r.size < min_leaf:
        return node
    node.left = _grow(z, t, x, s_l, e_l, depth + 1, max_depth, min_leaf, n_thr, rng)
    node.right = _grow(z, t, x, s_r, e_r, depth + 1, max_depth, min_leaf, n_thr, rng)
    return node


def _predict_leaf(node: _Node, row: FloatArray) -> _Node:
    while not node.is_leaf:
        node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[assignment]
    return node


def _collect_leaves(node: _Node, out: list[_Node]) -> None:
    if node.is_leaf:
        out.append(node)
    else:
        _collect_leaves(node.left, out)  # type: ignore[arg-type]
        _collect_leaves(node.right, out)  # type: ignore[arg-type]


def causal_tree(
    y: FloatArray,
    t: FloatArray,
    x: FloatArray,
    max_depth: int = 4,
    min_leaf: int = 20,
    honest: bool = True,
    seed: int = 0,
) -> dict[str, FloatArray | float | _Node]:
    """Grow one (honest) causal tree.

    Honest: a random half chooses splits, the complementary half is
    carried to leaves and used for effect estimates — the Athey-Imbens
    protocol. Returns the root node plus leaf effect stats.
    """
    y, t, x = _check_inputs(y, t, x)
    rng = np.random.default_rng(seed)
    z = _transform_outcome(y, t, x)
    idx = rng.permutation(y.size)
    if honest:
        split_idx = idx[: y.size // 2]
        est_idx = idx[y.size // 2 :]
    else:
        split_idx = est_idx = idx
    root = _grow(z, t, x, split_idx, est_idx, 0, max_depth, min_leaf, 8, rng)
    leaves: list[_Node] = []
    _collect_leaves(root, leaves)
    effects = np.array([_leaf_effect(z, t, lf.est_idx) for lf in leaves])
    return {
        "root": root,
        "n_leaves": float(len(leaves)),
        "effect_mean": float(effects.mean()),
        "effect_sd": float(effects.std()),
    }


def causal_forest(
    y: FloatArray,
    t: FloatArray,
    x: FloatArray,
    n_trees: int = 40,
    subsample: float = 0.5,
    max_depth: int = 4,
    min_leaf: int = 20,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    """Small honest causal forest.

    Each tree: honest split on a ``subsample`` share of rows.
    Prediction = mean leaf effect across trees whose leaf contains the
    point; ``se`` = across-tree std of that point's leaf effects /
    sqrt(B) (the infinitesimal-jackknife-style spread).
    Variable importance = fraction of internal splits on each feature.
    """
    y, t, x = _check_inputs(y, t, x)
    if not 5 <= n_trees <= 200:
        raise ValueError("n_trees in 5..200")
    rng = np.random.default_rng(seed)
    n = y.size
    z = _transform_outcome(y, t, x)
    per_tree_cate = np.empty((n_trees, n))
    split_counts = np.zeros(x.shape[1])
    splits_total = 0
    for b in range(n_trees):
        sub = rng.permutation(n)[: max(2 * min_leaf, int(subsample * n))]
        idx = rng.permutation(sub)
        split_idx = idx[: idx.size // 2]
        est_idx = idx[idx.size // 2 :]
        root = _grow(z, t, x, split_idx, est_idx, 0, max_depth, min_leaf, 8, rng)
        # variable importance: count internal splits per feature
        stack = [root]
        while stack:
            nd = stack.pop()
            if nd.is_leaf:
                continue
            split_counts[nd.feature] += 1
            splits_total += 1
            stack.extend([nd.left, nd.right])  # type: ignore[list-item]
        for i in range(n):
            lf = _predict_leaf(root, x[i])
            per_tree_cate[b, i] = _leaf_effect(z, t, lf.est_idx)
    cate = per_tree_cate.mean(axis=0)
    se = per_tree_cate.std(axis=0, ddof=1) / math.sqrt(n_trees)
    importance = split_counts / max(splits_total, 1)
    return {
        "cate": cate,
        "cate_se": se,
        "cate_mean": float(cate.mean()),
        "cate_sd": float(cate.std()),
        "importance": importance,
        "n_trees": float(n_trees),
        "splits_total": float(splits_total),
    }


def synth_cate(
    n: int = 1000,
    tau0: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC heterogeneous effect: ``tau(x) = tau0 + 1.2·1[x1>0]``
    on confounded observational data — treatment propensity tilts with
    x2."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 5))
    pr = 1.0 / (1.0 + np.exp(-0.8 * x[:, 1]))
    t = (rng.random(n) < pr).astype(np.float64)
    tau = tau0 + 1.2 * (x[:, 0] > 0)
    y = 0.8 * x[:, 1] + 0.5 * x[:, 2] + t * tau + rng.normal(0, 0.5, n)
    return {"y": y, "t": t, "x": x, "tau_true": tau, "tau_mean": np.float64(float(tau.mean()))}


def bench_causal_forest(seed: int = 20261231 + 182) -> dict[str, float]:
    """SYNTHETIC: forest recovers subgroup gap and ranks x1 top."""
    d = synth_cate(n=1200, seed=seed)
    y = np.asarray(d["y"])
    t = np.asarray(d["t"])
    x = np.asarray(d["x"])
    tau = np.asarray(d["tau_true"])
    f = causal_forest(y, t, x, n_trees=40, seed=seed)
    cate = np.asarray(f["cate"])
    imp = np.asarray(f["importance"])
    hi = x[:, 0] > 0
    gap_hat = float(cate[hi].mean() - cate[~hi].mean())
    gap_true = float(tau[hi].mean() - tau[~hi].mean())
    corr = float(np.corrcoef(cate, tau)[0, 1])
    f2 = causal_forest(y, t, x, n_trees=40, seed=seed)
    return {
        "synthetic_cate_corr": corr,
        "synthetic_gap_hat": gap_hat,
        "synthetic_gap_err": abs(gap_hat - gap_true),
        "synthetic_top_feature_is_x1": float(imp.argmax() == 0),
        "synthetic_ate_err": abs(float(f["cate_mean"]) - float(d["tau_mean"])),
        "synthetic_mean_se": float(np.mean(np.asarray(f["cate_se"]))),
        "synthetic_determinism": float(np.allclose(cate, np.asarray(f2["cate"]))),
    }
