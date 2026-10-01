"""Louvain-style modularity-maximizing community detection.

Weighted graphs built from correlation/similarity matrices admit
community structure (factor groups, sectors). This module implements
the Blondel et al. two-phase Louvain procedure — greedy local moves
that monotonically increase modularity, then graph aggregation — plus
Newman-Girvan modularity, ARI, and NMI cluster-quality scores.

``W`` is treated as a symmetric, nonnegative, hollow (zero-diagonal)
adjacency matrix; signed or non-hollow inputs are rejected (fail
closed) — sign-aware modularity (Gomez-Jensen-Arenas) is a deliberate
non-goal here.

Functions
---------
- :func:`modularity` — Newman-Girvan Q.
- :func:`louvain` — greedy Louvain → labels + Q + #communities.
- :func:`correlation_network` — corr matrix → nonnegative adjacency.
- :func:`planted_partition` — SBM adjacency generator.
- :func:`adjusted_rand`, :func:`nmi` — cluster-quality scores.
- :func:`bench_modularity_communities` — SYNTHETIC telemetry blob.

References
----------
- Blondel, Guillaume, Lambiotte & Lefebvre (2008). Fast unfolding of
  communities in large networks. *J. Stat. Mech.* — arXiv:0803.0476
  (verified).
- Newman & Girvan (2004). Finding and evaluating community structure
  in networks. *Phys. Rev. E* 69 (journal).
- Hubert & Arabie (1985). Comparing partitions. *J. Classification*
  (ARI; journal).
- Fortunato & Barthelemy (2007). Resolution limit in community
  detection. *PNAS* 104 — arXiv:physics/0607100 (verified).

Honesty
-------
Benches run on SYNTHETIC planted-partition and factor-correlation
networks only. Community detection on correlation matrices is a
structure diagnostic; never a diversification/PnL claim.

Composition notes
-----------------
- ``models/koopman_edmd.py`` (wave 23): spectral regime embeddings —
  this module is the discrete-community complement.
- ``models/tda_persistence.py`` (wave 25): topological regime features
  — communities are the discrete analog of H0 clusters.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _v_adj(W: FloatArray) -> FloatArray:
    w = np.asarray(W, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.isfinite(w).all():
        raise ValueError("adjacency must be finite")
    if not np.allclose(w, w.T, atol=1e-10):
        raise ValueError("adjacency must be symmetric")
    if np.abs(np.diag(w)).max() > 1e-10:
        raise ValueError("adjacency must be hollow (zero diagonal)")
    if (w < -1e-10).any():
        raise ValueError("adjacency must be nonnegative")
    return w


def modularity(W: FloatArray, labels: IntArray) -> float:
    """Newman-Girvan modularity ``Q = (1/2m) sum_ij [W_ij - k_i k_j/2m] d(c_i, c_j)``."""
    w = _v_adj(W)
    lab = np.asarray(labels, dtype=np.int64).ravel()
    if lab.size != w.shape[0]:
        raise ValueError("labels length must match adjacency size")
    if lab.size < 2:
        raise ValueError("need >= 2 nodes")
    k = w.sum(axis=1)
    m2 = float(k.sum())  # 2m
    if m2 <= 0.0:
        raise ValueError("empty graph has no modularity")
    q = 0.0
    for c in np.unique(lab):
        mask = lab == c
        e_c = float(w[np.ix_(mask, mask)].sum())
        vol_c = float(k[mask].sum())
        q += e_c / m2 - (vol_c / m2) ** 2
    return q


def louvain(
    W: FloatArray,
    seed: int = 0,
    max_passes: int = 100,
    tol: float = 1e-9,
) -> tuple[IntArray, float, int]:
    """Greedy Louvain community detection → ``(labels, Q, n_communities)``.

    Phase 1: iterate nodes in seeded random order; move each node into
    the neighboring community with the largest positive modularity
    gain ``dQ = (in_weight_delta - k_i * vol_c / m)`` (m = total edge
    weight). Phase 2: aggregate communities into super-nodes and
    repeat until no pass improves Q beyond ``tol``.
    """
    w0 = _v_adj(W)
    n0 = w0.shape[0]
    if n0 < 2:
        raise ValueError("need >= 2 nodes")
    rng = np.random.default_rng(seed)
    # node -> original-index set; labels at level 0 are identity
    node_members: list[list[int]] = [[i] for i in range(n0)]
    w = w0.copy()

    total_q_gain = 0.0
    for _pass in range(max_passes):
        m2 = float(w.sum())
        if m2 <= 0:
            break
        k_deg = w.sum(axis=1)
        n = w.shape[0]
        # comm_of: current community per (super)node — start singletons
        comm_of = np.arange(n)
        # community volumes
        vol = k_deg.copy()
        improved = False
        moved_any = True
        sweeps = 0
        while moved_any and sweeps < 50:
            sweeps += 1
            moved_any = False
            for i in rng.permutation(n):
                ci = comm_of[i]
                # weight from i to each community (self-loops excluded)
                w_to = np.zeros(n)
                nbrs = np.nonzero(w[i])[0]
                for j in nbrs:
                    if j == i:
                        continue  # self-loops are not community edges
                    w_to[comm_of[j]] += w[i, j]
                # drop-in gain of moving i into c vs staying in ci:
                #   gain_c = w_to_c - k_i * vol_c / m2
                #   stay   = w_to_ci - k_i * (vol_ci - k_i) / m2
                gain = w_to - k_deg[i] * vol / m2
                stay = w_to[ci] - k_deg[i] * (vol[ci] - k_deg[i]) / m2
                gain[ci] = -np.inf  # argmax over destinations only
                c_best = int(np.argmax(gain))
                if np.isfinite(gain[c_best]) and gain[c_best] > stay + tol:
                    comm_of[i] = c_best
                    vol[ci] -= k_deg[i]
                    vol[c_best] += k_deg[i]
                    moved_any = True
                    improved = True
        if not improved:
            break
        total_q_gain += 1.0  # monotone-in-Q pass counter (bounded)
        # phase 2: aggregate communities
        uniq = np.unique(comm_of)
        remap = {c: r for r, c in enumerate(uniq)}
        m_new = uniq.size
        if m_new == n:
            break
        w_new = np.zeros((m_new, m_new))
        for i in range(n):
            for j in range(n):
                w_new[remap[comm_of[i]], remap[comm_of[j]]] += w[i, j]
        # internal weight is preserved as self-loops on super-nodes —
        # dropping it collapses every community (diagonal is part of the
        # degree/volume bookkeeping at the next level)
        w_new *= 1.0
        # fold original membership
        members_new: list[list[int]] = [[] for _ in range(m_new)]
        for i in range(n):
            members_new[remap[comm_of[i]]].extend(node_members[i])
        node_members = members_new
        w = w_new
    # expand labels back to original nodes
    labels = np.empty(n0, dtype=np.int64)
    for c_id, members in enumerate(node_members):
        for orig in members:
            labels[orig] = c_id
    n_comm = int(labels.max() + 1) if labels.size else 0
    return labels, modularity(w0, labels), n_comm


def correlation_network(corr: FloatArray, threshold: float | None = None) -> FloatArray:
    """Correlation matrix → nonnegative hollow adjacency.

    ``W = clip(corr, 0, 1)`` (positive-correlation network); if
    ``threshold`` is given, edges below it are zeroed. Negative weights
    are dropped — Louvain here is nonnegative-only (documented scope).
    """
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be square")
    if not np.isfinite(c).all():
        raise ValueError("corr must be finite")
    if not np.allclose(c, c.T, atol=1e-8):
        raise ValueError("corr must be symmetric")
    w = np.clip(c, 0.0, None)
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        w = np.where(w >= threshold, w, 0.0)
    np.fill_diagonal(w, 0.0)
    return np.asarray(w, dtype=np.float64)


def planted_partition(
    n: int,
    k: int,
    p_in: float,
    p_out: float,
    seed: int = 0,
) -> tuple[FloatArray, IntArray]:
    """SBM adjacency: ``(W, labels_true)``, binary weights, hollow."""
    if n < k or k < 2 or n < 4:
        raise ValueError("need n >= k >= 2 and n >= 4")
    if not (0.0 < p_out < p_in <= 1.0):
        raise ValueError("need 0 < p_out < p_in <= 1")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(k), n // k)
    if labels.size < n:
        labels = np.concatenate([labels, rng.integers(0, k, n - labels.size)])
    rng.shuffle(labels)
    w = np.zeros((n, n))
    same = labels[:, None] == labels[None, :]
    probs = np.where(same, p_in, p_out)
    mask = rng.random((n, n)) < probs
    w = np.triu(mask, 1).astype(float)
    w = w + w.T
    return np.asarray(w, dtype=np.float64), labels.astype(np.int64)


def adjusted_rand(labels_true: IntArray, labels_est: IntArray) -> float:
    """Adjusted Rand index (Hubert-Arabie 1985)."""
    a = np.asarray(labels_true, dtype=np.int64).ravel()
    b = np.asarray(labels_est, dtype=np.int64).ravel()
    if a.size != b.size or a.size < 2:
        raise ValueError("labels must match and have >= 2 elements")
    n = a.size
    classes_a = np.unique(a)
    classes_b = np.unique(b)
    nij = np.zeros((classes_a.size, classes_b.size))
    for i, ca in enumerate(classes_a):
        for j, cb in enumerate(classes_b):
            nij[i, j] = float(((a == ca) & (b == cb)).sum())

    def comb2_arr(x: FloatArray) -> FloatArray:
        return x * (x - 1.0) / 2.0

    sum_ij = float(comb2_arr(nij).sum())
    sum_a = float(comb2_arr(nij.sum(axis=1)).sum())
    sum_b = float(comb2_arr(nij.sum(axis=0)).sum())
    total = float(n) * (float(n) - 1.0) / 2.0
    expected = sum_a * sum_b / total if total > 0 else 0.0
    maxi = 0.5 * (sum_a + sum_b)
    denom = maxi - expected
    return (sum_ij - expected) / denom if denom > 0 else 1.0


def nmi(labels_true: IntArray, labels_est: IntArray) -> float:
    """Normalized mutual information (arithmetic mean normalization)."""
    a = np.asarray(labels_true, dtype=np.int64).ravel()
    b = np.asarray(labels_est, dtype=np.int64).ravel()
    if a.size != b.size or a.size < 2:
        raise ValueError("labels must match and have >= 2 elements")
    n = a.size
    ca = np.unique(a)
    cb = np.unique(b)
    nij = np.zeros((ca.size, cb.size))
    for i, x in enumerate(ca):
        for j, y in enumerate(cb):
            nij[i, j] = float(((a == x) & (b == y)).sum())
    pi = nij / n
    pa = pi.sum(axis=1)
    pb = pi.sum(axis=0)
    mi = 0.0
    for i in range(ca.size):
        for j in range(cb.size):
            if pi[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                mi += pi[i, j] * math.log(pi[i, j] / (pa[i] * pb[j]))
    ha = -float((pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = -float((pb[pb > 0] * np.log(pb[pb > 0])).sum())
    denom = 0.5 * (ha + hb)
    return mi / denom if denom > 0 else 1.0


def _factor_corr(n: int, k: int, loading: float, seed: int) -> tuple[FloatArray, IntArray]:
    """Factor-structured correlation matrix + true group labels."""
    rng = np.random.default_rng(seed)
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    labels = np.repeat(np.arange(k), sizes)
    L = np.zeros((n, k))
    for i, c in enumerate(labels):
        L[i, c] = loading + 0.1 * rng.standard_normal()
    C = L @ L.T + np.diag(1.0 - np.diag(L @ L.T))
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    return np.asarray(corr, dtype=np.float64), labels.astype(np.int64)


def bench_modularity_communities(seed: int = 20260206) -> dict[str, float]:
    """SYNTHETIC bench: recovery on planted partitions + corr networks."""
    out: dict[str, float] = {}
    # SBM recovery
    w, truth = planted_partition(60, 4, p_in=0.5, p_out=0.05, seed=seed)
    labels, q, nc = louvain(w, seed=seed)
    out["synthetic_ari"] = adjusted_rand(truth, labels)
    out["synthetic_nmi"] = nmi(truth, labels)
    out["synthetic_k_err"] = float(abs(nc - 4)) / 4.0
    out["synthetic_q"] = q
    # Q improved over singleton/everything baseline
    q_singleton = modularity(w, np.arange(60))
    q_one = modularity(w, np.zeros(60, dtype=np.int64))
    out["synthetic_q_gain_singleton"] = q - q_singleton
    out["synthetic_q_gain_one"] = q - q_one
    # factor-structured correlation network
    corr, groups = _factor_corr(48, 3, loading=0.8, seed=seed + 1)
    wn = correlation_network(corr)
    lab2, q2, nc2 = louvain(wn, seed=seed + 1)
    out["synthetic_corr_ari"] = adjusted_rand(groups, lab2)
    out["synthetic_corr_q"] = q2
    # determinism
    l1, q1a, _ = louvain(w, seed=seed)
    l2, q1b, _ = louvain(w, seed=seed)
    out["synthetic_determinism"] = float(np.array_equal(l1, l2) and q1a == q1b)
    return out
