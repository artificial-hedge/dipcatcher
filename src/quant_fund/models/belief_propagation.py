"""Belief propagation canon: sum-product message passing on trees (exact) (SYNTHETIC)
and loopy pairwise MRFs (approximate, damping), plus Bethe free-energy
marginals. ``bench_belief_propagation`` uses a chain (exact = forward/
backward) and a small loopy grid where BP marginals are compared to the
brute-force joint distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _norm(v: FloatArray) -> FloatArray:
    s = v.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / v.size)
    return v / s


def bp_chain(pot_node: FloatArray, pot_edge: FloatArray, it: int = 400) -> FloatArray:
    """Exact marginals on a length-n chain. pot_node (n,k); pot_edge
    (n-1,k,k) with pot_edge[i][a,b] = psi(x_i=a, x_{i+1}=b)."""
    pot_node = np.asarray(pot_node, dtype=np.float64)
    pot_edge = np.asarray(pot_edge, dtype=np.float64)
    n, k = pot_node.shape
    if pot_edge.shape != (n - 1, k, k):
        raise ValueError("edge shape mismatch")
    fwd = np.ones((n, k))
    bwd = np.ones((n, k))
    for _ in range(it):
        for i in range(1, n):
            m = (pot_node[i - 1] * fwd[i - 1])[:, None] * pot_edge[i - 1]
            fwd[i] = _norm(m.sum(axis=0))
        for i in range(n - 2, -1, -1):
            m = (pot_node[i + 1] * bwd[i + 1])[None, :] * pot_edge[i]
            bwd[i] = _norm(m.sum(axis=1))
    return np.stack([_norm(pot_node[i] * fwd[i] * bwd[i]) for i in range(n)])


def bp_loopy(
    pot_node: FloatArray,
    edges: list[tuple[int, int]],
    pot_edge: FloatArray,
    it: int = 200,
    damp: float = 0.5,
) -> FloatArray:
    """Loopy BP on a pairwise MRF. pot_edge[e][a,b] couples edge e's
    (i,j). Returns approximate node marginals."""
    pot_node = np.asarray(pot_node, dtype=np.float64)
    pot_edge = np.asarray(pot_edge, dtype=np.float64)
    n, k = pot_node.shape
    msgs: dict[tuple[int, int], FloatArray] = {}
    for i, j in edges:
        msgs[(i, j)] = np.full(k, 1.0 / k)
        msgs[(j, i)] = np.full(k, 1.0 / k)
    edge_of = {(i, j): e for e, (i, j) in enumerate(edges)}
    edge_of.update({(j, i): e for e, (i, j) in enumerate(edges)})
    for _ in range(it):
        new: dict[tuple[int, int], FloatArray] = {}
        for i, j in msgs:
            # message i->j
            e = edge_of[(i, j)]
            pe = pot_edge[e]
            if edges[e][0] != i:
                pe = pe.T
            inc = np.ones(k)
            for u, v2 in msgs:
                if v2 == i and u != j:
                    inc *= msgs[(u, i)]
            f = (pot_node[i] * inc)[:, None] * pe
            m = _norm(f.sum(axis=0))
            new[(i, j)] = (1 - damp) * msgs[(i, j)] + damp * m
        diff = max(float(np.max(np.abs(new[kk] - msgs[kk]))) for kk in msgs)
        msgs = new
        if diff < 1e-9:
            break
    out = np.zeros((n, k))
    for i in range(n):
        inc = np.ones(k)
        for u, v2 in msgs:
            if v2 == i:
                inc *= msgs[(u, i)]
        out[i] = _norm(pot_node[i] * inc)
    return out


def brute_marginals(
    pot_node: FloatArray,
    edges: list[tuple[int, int]],
    pot_edge: FloatArray,
) -> FloatArray:
    """Exact marginals by enumerating the joint (small graphs only)."""
    pot_node = np.asarray(pot_node, dtype=np.float64)
    n, k = pot_node.shape
    joint = np.ones([k] * n)
    for i in range(n):
        joint = joint * pot_node[i].reshape([k if ax == i else 1 for ax in range(n)])
    for e, (i, j) in enumerate(edges):
        gi = np.arange(k).reshape([k if ax == i else 1 for ax in range(n)])
        gj = np.arange(k).reshape([k if ax == j else 1 for ax in range(n)])
        joint = joint * pot_edge[e][gi, gj]
    joint /= joint.sum()
    marg = np.zeros((n, k))
    for i in range(n):
        axes = tuple(ax for ax in range(n) if ax != i)
        marg[i] = joint.sum(axis=axes)
    return marg


def bench_belief_propagation(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    k = 3
    # chain of 8 nodes with mixed attractive/repulsive edges
    n = 8
    pn = np.exp(0.4 * rng.standard_normal((n, k)))
    pn = pn / pn.sum(axis=1, keepdims=True)
    pe = np.exp(0.6 * rng.standard_normal((n - 1, k, k)))
    pe += 0.5 * np.eye(k)[None]  # mild attraction
    bp = bp_chain(pn, pe)
    edges_ch = [(i, i + 1) for i in range(n - 1)]
    exact = brute_marginals(pn, edges_ch, pe)
    # loopy: 6-node ring + one chord
    n2 = 6
    pn2 = np.exp(0.4 * rng.standard_normal((n2, k)))
    pn2 = pn2 / pn2.sum(axis=1, keepdims=True)
    edges2 = [(i, (i + 1) % n2) for i in range(n2)] + [(0, 3)]
    pe2 = np.exp(0.5 * rng.standard_normal((len(edges2), k, k))) + 0.4 * np.eye(k)[None]
    bp2 = bp_loopy(pn2, edges2, pe2)
    exact2 = brute_marginals(pn2, edges2, pe2)
    return {
        "synthetic_bp_chain_l1": float(np.abs(bp - exact).max()),
        "synthetic_bp_loopy_l1": float(np.abs(bp2 - exact2).max()),
        "synthetic_bp_chain_simplex_err": float(np.abs(bp.sum(1) - 1.0).max()),
        "synthetic_bp_loopy_simplex_err": float(np.abs(bp2.sum(1) - 1.0).max()),
    }
