"""Shared synthetic geometry fixtures (wave 140).

- `synth_hierarchy`: binary-tree leaf embeddings with tree-metric
  distances — hyperbolic geometry fits them with low distortion where
  Euclidean needs high dimension.
- `synth_partwhole`: each sample combines a "part" pair whose pattern
  determines the class — dynamic routing pools part votes.
- `synth_field`: smooth 2-D field f(x,y) = sin/cos mixture + its gradient —
  INRs fit the field and gradients jointly.
- `synth_rot_cloud`: 3-D point clouds whose energy depends only on
  rotation-invariant pairwise distances.
- `synth_monotonic`: noisy monotone map y = sigmoid-like ramp.
- `synth_sort`: vectors to be ranked — differentiable sort recovers rank.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def synth_hierarchy(
    depth: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Leaves of a binary tree of given depth; returns leaf vectors,
    leaf indices, and the true tree distance matrix."""
    n_leaves = 2**depth
    vecs = np.zeros((n_leaves, 2))
    for i in range(n_leaves):
        theta = 2 * np.pi * i / n_leaves
        vecs[i] = np.array([np.cos(theta), np.sin(theta)])
    dist = np.zeros((n_leaves, n_leaves))
    for i in range(n_leaves):
        for j in range(n_leaves):
            li = bin(i)[2:].zfill(depth)
            lj = bin(j)[2:].zfill(depth)
            shared = next((k for k in range(depth) if li[k] != lj[k]), depth)
            dist[i, j] = 2 * (depth - shared)
    vecs += 0.05 * rng.standard_normal(vecs.shape)
    return vecs, np.arange(n_leaves, dtype=np.int64), dist


def synth_partwhole(n: int, rng: np.random.Generator) -> tuple[FloatArray, NDArray[np.int64]]:
    """4 slots of 4-dim part features; class = whether parts 0,1 point
    the same way (dot sign). Capsule routing must pool part evidence."""
    n_parts = 4
    d_part = 4
    x = rng.standard_normal((n, n_parts, d_part))
    same = (x[:, 0] * x[:, 1]).sum(-1) > 0
    y = same.astype(np.int64)
    x += rng.normal(0, 0.1, x.shape)
    return x, y


def synth_field(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Smooth field f(x,y)=sin(2x)cos(3y)+0.5x; returns pts, f, grad."""
    pts = rng.uniform(-np.pi, np.pi, (n, 2))
    f = np.sin(2 * pts[:, 0]) * np.cos(3 * pts[:, 1]) + 0.5 * pts[:, 0]
    grad = np.stack(
        [
            2 * np.cos(2 * pts[:, 0]) * np.cos(3 * pts[:, 1]) + 0.5,
            -3 * np.sin(2 * pts[:, 0]) * np.sin(3 * pts[:, 1]),
        ],
        -1,
    )
    return pts, f[:, None], grad


def synth_rot_cloud(
    n: int, m_pts: int, rng: np.random.Generator
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Point clouds; label = mean pairwise distance above threshold —
    invariant to global rotation. Test set can be freely rotated."""
    x = rng.normal(0, 1.0, (n, m_pts, 3))
    spread = rng.uniform(0.4, 1.6, (n, 1, 1))
    x = x * spread
    d = np.sqrt(((x[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1)).mean((1, 2))
    y = (d > np.median(d)).astype(np.int64)
    return x, y


def synth_monotonic(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    x = rng.uniform(-3, 3, (n, 1))
    y = np.tanh(x[:, 0]) * 0.8 + 0.25 * rng.standard_normal(n)
    return x, y[:, None]


def synth_sort(n: int, m: int, rng: np.random.Generator) -> tuple[FloatArray, NDArray[np.int64]]:
    x = rng.uniform(0, 1, (n, m))
    ranks = x.argsort(-1).argsort(-1).astype(np.int64)
    return x, ranks
