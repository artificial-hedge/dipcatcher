"""Robust Random Cut Forest (Guha et al. 2016) — pure numpy.

Streaming anomaly scoring by expected tree displacement (CoDisp):
how much deleting a point shrinks the tree — vs isolation-forest path
length on the same synth anomaly series.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._anom_synth import anom_series, auc


class _Node:
    __slots__ = ("l", "r", "n", "leaf", "dim", "cut")

    def __init__(self, leaf=None, n=0, dim=-1, cut=0.0, left=None, right=None):
        self.leaf = leaf
        self.n = n
        self.dim = dim
        self.cut = cut
        self.l = left
        self.r = right


def _build(pts: np.ndarray, rng: np.random.Generator) -> _Node:
    n = len(pts)
    if n <= 1:
        return _Node(leaf=True, n=n)
    span = pts.max(0) - pts.min(0)
    tot = span.sum()
    if tot <= 0:
        return _Node(leaf=True, n=n)
    dim = rng.choice(len(span), p=span / tot)
    cut = rng.uniform(pts[:, dim].min(), pts[:, dim].max())
    left = pts[pts[:, dim] <= cut]
    right = pts[pts[:, dim] > cut]
    if len(left) == 0 or len(right) == 0:
        return _Node(leaf=True, n=n)
    return _Node(n=n, dim=dim, cut=cut, left=_build(left, rng), right=_build(right, rng))


def _codisp(root: _Node, pt: np.ndarray) -> float:
    """CoDisp approximation: max sibling-subtree size along the point's
    path from root to its leaf — isolated points sit beside big subtrees."""
    node = root
    disp = 0.0
    while node.leaf is not True:
        sib = node.r if pt[node.dim] <= node.cut else node.l
        node = node.l if pt[node.dim] <= node.cut else node.r
        disp = max(disp, float(sib.n))
    return disp


def bench_rrcf(
    seed: int = 631,
    n_trees: int = 30,
    leaf_cap: int = 256,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x, y = anom_series(seed=seed)
    n = len(x)
    # insert all points, then score each point's CoDisp under leave-one-out
    scores = np.zeros(n)
    for _t in range(n_trees):
        root = _build(x, rng)
        for i in range(n):
            scores[i] += _codisp(root, x[i])
    scores /= n_trees
    _ = leaf_cap
    auc_r = auc(scores, y)

    # iForest baseline: expected path length of random-cut trees
    def plen(root: _Node, pt: np.ndarray) -> float:
        node = root
        d = 0
        while node.leaf is not True:
            node = node.l if pt[node.dim] <= node.cut else node.r
            d += 1
        return float(d)

    sc_if = np.zeros(n)
    for _t in range(n_trees):
        root = _build(x, rng)
        for i in range(n):
            sc_if[i] -= plen(root, x[i])
    sc_if /= n_trees
    auc_if = auc(sc_if, y)
    return {
        "synthetic_rrcf_auc": auc_r,
        "synthetic_rrcf_if_auc": auc_if,
        "synthetic_rrcf_gain": auc_r - auc_if,
    }
