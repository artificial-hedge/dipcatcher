"""Link-analysis topology: PageRank power iteration with
dangling-node correction and teleport personalization, HITS
hub/authority scores, and conductance of a vertex subset.
Synthetic bench gates planted-hub ranking."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _check_adj(adj: FloatArray) -> FloatArray:
    adj = np.asarray(adj, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square")
    return adj


def pagerank(
    adj: FloatArray,
    alpha: float = 0.85,
    pers: FloatArray | None = None,
    it: int = 200,
    tol: float = 1e-10,
) -> FloatArray:
    """PageRank (Brin & Page 1998): r = αP'r + (1−α)v +
    dangling-mass redistribution into v."""
    adj = _check_adj(adj)
    n = adj.shape[0]
    if pers is None:
        pers = np.full(n, 1.0 / n)
    pers = np.asarray(pers, dtype=np.float64)
    pers = pers / pers.sum()
    outdeg = adj.sum(axis=1)
    p = adj / np.maximum(outdeg, 1e-12)[:, None]
    dangling = (outdeg == 0).astype(np.float64)
    r = np.full(n, 1.0 / n)
    for _ in range(it):
        r_new = alpha * (r @ p + (r @ dangling) * pers) + (1 - alpha) * pers
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return np.asarray(r / r.sum())


def hits(adj: FloatArray, it: int = 100, tol: float = 1e-10) -> dict[str, FloatArray]:
    """HITS (Kleinberg 1999): authorities = A'hubs,
    hubs = A·authorities, L2-normalized."""
    adj = _check_adj(adj)
    n = adj.shape[0]
    a = np.ones(n)
    h = np.ones(n)
    for _ in range(it):
        a_new = adj.T @ h
        h_new = adj @ a_new
        a_new /= max(np.linalg.norm(a_new), 1e-16)
        h_new /= max(np.linalg.norm(h_new), 1e-16)
        if np.abs(a_new - a).sum() < tol and np.abs(h_new - h).sum() < tol:
            a, h = a_new, h_new
            break
        a, h = a_new, h_new
    return {"authority": a, "hub": h}


def conductance(adj: FloatArray, s: BoolArray) -> float:
    """Φ(S) = cut(S,S̄) / min(vol(S), vol(S̄))."""
    adj = _check_adj(adj)
    mask = np.asarray(s, dtype=bool)
    cut = float(adj[np.ix_(mask, ~mask)].sum())
    vol_s = float(adj[np.ix_(mask, mask)].sum()) + cut
    vol_c = float(adj[np.ix_(~mask, ~mask)].sum()) + cut
    denom = min(vol_s, vol_c)
    return float(cut / denom) if denom > 0 else 0.0


def spectral_bisect(adj: FloatArray) -> FloatArray:
    """Fiedler-vector sign bipartition of the normalized
    Laplacian."""
    adj = _check_adj(adj)
    d = adj.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    l_sym = np.eye(len(d)) - dinv[:, None] * adj * dinv[None, :]
    vals, vecs = np.linalg.eigh(l_sym)
    # Connected graph: sign of the second-smallest eigenvector.
    # Disconnected graph: λ0 has multiplicity = #components, so the
    # eigensolver returns an *arbitrary* basis of the zero eigenspace —
    # any single column may be non-separating (BLAS-dependent). The
    # component indicators instead live in that whole eigenspace: each
    # node's row of coordinates lies on a per-component ray, so cluster
    # the unit rays against the two most-separated directions.
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    if len(vals) >= 2 and vals[1] <= 1e-8:
        coords = vecs[:, : max(2, int(np.sum(vals <= 1e-8)))]
        norms = np.maximum(np.linalg.norm(coords, axis=1), 1e-12)
        u = coords / norms[:, None]
        ref0 = u[0]
        ref1 = u[int(np.argmin(u @ ref0))]
        nearer = np.abs(u @ ref1) > np.abs(u @ ref0)
        return np.asarray(nearer.astype(np.int64))
    fiedler = vecs[:, 1]
    return np.asarray((fiedler > 0).astype(np.int64))


def bench_pagerank(seed: int = 544) -> dict[str, float]:
    """SYNTHETIC: directed hub planted in a 3-community
    ring — PageRank must rank the hub top; HITS hub/
    authority split consistent; conductance of a true
    community beats a random split."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n, k, per = 90, 3, 30
    adj = np.zeros((n, n))
    for c in range(k):
        idx = np.arange(c * per, (c + 1) * per)
        # dense intra-community ring
        for i in idx:
            for j in idx:
                if i != j and rng.uniform() < 0.35:
                    adj[i, j] = 1
    # planted hub: every node links to node 0 with p .9
    hub = 0
    for ii in range(n):
        if ii != hub and rng.uniform() < 0.9:
            adj[ii, hub] = 1
    pr = pagerank(adj)
    out["synthetic_pr_hub_mass"] = float(pr[hub])
    if int(np.argmax(pr)) != hub:
        raise ValueError(f"pagerank missed hub: top={int(np.argmax(pr))}")
    h = hits(adj)
    auth = np.asarray(h["authority"])
    out["synthetic_hits_hub_auth"] = float(auth[hub])
    if int(np.argmax(auth)) != hub:
        raise ValueError(f"hits missed hub: top={int(np.argmax(auth))}")
    # conductance: true community vs random half
    s_true = np.zeros(n, dtype=bool)
    s_true[:per] = True
    phi_t = conductance(adj, s_true)
    s_rand = rng.uniform(size=n) < 0.5
    phi_r = conductance(adj, s_rand)
    out["synthetic_conductance_comm"] = phi_t
    out["synthetic_conductance_rand"] = phi_r
    if phi_t >= phi_r:
        raise ValueError(f"conductance off: {phi_t} vs {phi_r}")
    return out
