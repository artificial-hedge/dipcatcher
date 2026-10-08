"""Isolation forest anomaly detection (Liu, Ting & Zhou 2008) (SYNTHETIC).

Each iTree recursively partitions the feature space on a random
feature + random split point; anomalies isolate in few splits.
For a subsample size psi the expected path length of an
unsuccessful BST search is

    c(psi) = 2 (ln(psi - 1) + gamma_E) - 2 (psi - 1) / psi

and the anomaly score is s = 2^{-E[h(x)] / c(psi)}: near 1 for
anomalies, ~0.5 for ordinary points. We also implement the
extended iForest rotation (Hariri, Kind & Brunner 2019) where
splits use a random hyperplane through a random subsample
normal instead of axis-aligned cuts — handles elongated
clusters the axis-aligned variant scores poorly.

Honesty: unsupervised scoring only — no calibrated false-
positive rates; scores are monotone in isolation depth, not
posterior probabilities. The bench plants a tight cluster plus
scattered outliers and checks the outliers occupy the top
decile of scores. Fail-closed on n < 8, constant features,
or psi < 2.

References: Liu, Ting & Zhou (2008) ICDM:413; Liu, Ting &
Zhou (2012) TKDD 6:3; Hariri, Kind & Brunner (2019) ICDM;
Emmott et al. (2013) Numenta benchmark (scoring context).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class _Node:
    leaf: bool
    size: int
    depth: int
    j: int = -1
    normal: FloatArray | None = None
    cut: float = 0.0
    left: _Node | None = None
    right: _Node | None = None


def _c_factor(psi: float) -> float:
    if psi <= 1.0:
        return 1.0
    if psi <= 2.0:
        return 1.0
    g = np.euler_gamma
    return float(2.0 * (np.log(psi - 1.0) + g) - 2.0 * (psi - 1.0) / psi)


def _build_tree(
    x: FloatArray,
    depth: int,
    depth_cap: int,
    rng: np.random.Generator,
    extended: bool,
) -> _Node:
    n = x.shape[0]
    if n <= 1 or depth >= depth_cap:
        return _Node(leaf=True, size=n, depth=depth)
    if extended:
        # random unit normal + random intercept within data span
        normal = np.asarray(rng.standard_normal((x.shape[1],)), dtype=np.float64)
        nn = float(np.linalg.norm(normal))
        if nn <= 1e-12:
            return _Node(leaf=True, size=n, depth=depth)
        normal /= nn
        proj = x @ normal
        lo, hi = float(proj.min()), float(proj.max())
        if hi - lo <= 1e-12:
            return _Node(leaf=True, size=n, depth=depth)
        cut = float(rng.uniform(lo, hi))
        mask = proj <= cut
        return _Node(
            leaf=False,
            size=n,
            depth=depth,
            normal=normal,
            cut=cut,
            left=_build_tree(x[mask], depth + 1, depth_cap, rng, extended),
            right=_build_tree(x[~mask], depth + 1, depth_cap, rng, extended),
        )
    spread = np.ptp(x, axis=0)
    if float(spread.max()) <= 1e-12:
        return _Node(leaf=True, size=n, depth=depth)
    cols = np.flatnonzero(spread > 1e-12)
    j = int(rng.choice(cols))
    lo, hi = float(x[:, j].min()), float(x[:, j].max())
    cut = float(rng.uniform(lo, hi))
    mask = x[:, j] <= cut
    if mask.all() or (~mask).all():
        return _Node(leaf=True, size=n, depth=depth)
    return _Node(
        leaf=False,
        size=n,
        depth=depth,
        j=j,
        cut=cut,
        left=_build_tree(x[mask], depth + 1, depth_cap, rng, extended),
        right=_build_tree(x[~mask], depth + 1, depth_cap, rng, extended),
    )


def _path_length(tree: _Node, row: FloatArray) -> float:
    if tree.leaf:
        return float(tree.depth) + _c_factor(float(tree.size))
    if tree.normal is not None:
        side = float(row @ tree.normal) <= tree.cut
    else:
        side = float(row[tree.j]) <= tree.cut
    nxt = tree.left if side else tree.right
    if not (nxt is not None):
        raise ValueError("nxt is not None")
    return _path_length(nxt, row)


def isolation_forest(
    x: FloatArray,
    n_trees: int = 200,
    psi: int = 256,
    extended: bool = False,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Anomaly scores s in (0,1] — higher = more anomalous."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    n = xx.shape[0]
    if n < 8:
        raise ValueError("need n>=8")
    if float(np.ptp(xx, axis=0).max()) <= 1e-12:
        raise ValueError("constant features")
    sub = int(min(psi, n))
    if sub < 2:
        raise ValueError("psi<2")
    rng = np.random.default_rng(seed)
    cap = int(np.ceil(np.log2(max(sub, 2))))
    paths = np.zeros(n)
    for _ in range(n_trees):
        idx = rng.choice(n, size=sub, replace=False)
        tree = _build_tree(xx[idx], 0, cap, rng, extended)
        for i in range(n):
            paths[i] += _path_length(tree, xx[i])
    paths /= n_trees
    c = _c_factor(float(sub))
    s = np.power(2.0, -paths / max(c, 1e-12))
    return {
        "scores": np.asarray(s, dtype=np.float64),
        "mean_score": float(s.mean()),
        "c_factor": c,
    }


def bench_isolation_forest(seed: int = 20261231 + 453) -> dict[str, float]:
    """SYNTHETIC check — planted outliers occupy top scores."""
    rng = np.random.default_rng(seed)
    n_in, n_out = 300, 10
    core = rng.normal(scale=1.0, size=(n_in, 2))
    outliers = rng.uniform(-9, 9, size=(n_out, 2))
    x = np.vstack([core, outliers])
    out = isolation_forest(x, n_trees=300, psi=128, seed=seed)
    s = np.asarray(out["scores"], dtype=np.float64)
    order = np.argsort(-s)
    top_set = set(order[:n_out].tolist())
    planted = set(range(n_in, n_in + n_out))
    hits = len(top_set & planted)
    mean_in = float(s[:n_in].mean())
    mean_out = float(s[n_in:].mean())
    if hits < 6 or mean_out <= mean_in:
        raise ValueError(
            f"iforest off: hits={hits}/{n_out} s_out={mean_out:.3f} s_in={mean_in:.3f}"
        )
    return {
        "synthetic_top_hits": float(hits),
        "synthetic_score_gap": mean_out - mean_in,
        "synthetic_score": 1.0,
    }
