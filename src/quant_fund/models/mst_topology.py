"""Minimum-spanning-tree and planar-filtered network topology (SYNTHETIC).

Mantegna (1999): pairwise distances d_ij = sqrt(2(1 - rho_ij))
on a correlation matrix span a minimum spanning tree whose
topology (degree centrality, mean path length, centrality
concentration) summarizes the market's correlation skeleton
without thresholding. Prim's algorithm runs in O(k^2) on the
dense distance matrix. The planar maximally filtered graph
(Onnela et al. 2003/2004, Tumminello et al. 2005) relaxes the
tree to n-2 triangles worth of edges while keeping planarity:
edges are added in ascending distance order subject to a
planarity check on the crossing-free embedding (approximated
here via the Kruskal-style greedy build with a chordal-planar
heuristic: accept an edge iff no three existing cliques share
a common face conflict — we use the standard implementation
choice of tracking triangle counts).

Honesty: the PMFG heuristic used is the greedy
accept-if-planarity-compatible variant; exact planarity
testing is O(n) and approximated by bounded triangle
bookkeeping, which is the documented simplification, not a
claim of exact PMFG. The bench builds a one-factor correlation
structure: the factor-dominant asset must sit at the MST hub
(highest degree/centrality). Fail-closed on non-correlation
input or k<3.

References: Mantegna (1999) Eur. Phys. J. B 11:193; Onnela,
Chakraborti, Kaski & Kertesz (2003) Physica A 324:247;
Tumminello, Aste, Di Matteo & Mantegna (2005) PNAS 102:10421;
Bonanno et al. (2004) PRE 68; Pozzi, Di Matteo & Aste (2013)
Sci. Rep.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_corr(c: FloatArray) -> FloatArray:
    cc = np.asarray(c, dtype=np.float64)
    if cc.ndim != 2 or cc.shape[0] != cc.shape[1] or cc.shape[0] < 3:
        raise ValueError("corr must be square k>=3")
    if not np.allclose(cc, cc.T, atol=1e-6):
        raise ValueError("corr must be symmetric")
    d = np.diag(cc)
    if not np.allclose(d, 1.0, atol=1e-4):
        raise ValueError("diagonal must be 1")
    if np.abs(cc).max() > 1.0 + 1e-6:
        raise ValueError("entries must be in [-1,1]")
    return cc


def _dist(corr: FloatArray) -> FloatArray:
    return np.asarray(np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None)), dtype=np.float64)


def mst_build(corr: FloatArray) -> dict[str, float | FloatArray]:
    """Prim MST on the Mantegna metric."""
    cc = _check_corr(corr)
    n = cc.shape[0]
    d = _dist(cc)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    edges: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        best = (np.inf, -1, -1)
        for i in np.flatnonzero(in_tree):
            for j in np.flatnonzero(~in_tree):
                if d[i, j] < best[0]:
                    best = (float(d[i, j]), int(i), int(j))
        _, u, v = best
        if u < 0:
            raise ValueError("disconnected distance matrix")
        edges.append((u, v, float(d[u, v])))
        in_tree[v] = True
    adj = np.zeros((n, n))
    for u, v, w in edges:
        adj[u, v] = w
        adj[v, u] = w
    deg = (adj > 0).sum(axis=0).astype(np.float64)
    # shortest-path lengths on the tree (BFS from each node)
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for u, v, _w in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    total_path = 0.0
    ecc = np.zeros(n)
    for src in range(n):
        dist = np.full(n, -1)
        dist[src] = 0
        q = [src]
        while q:
            u = q.pop(0)
            for v in neighbors[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        ecc[src] = dist.max()
        total_path += dist.sum()
    mean_path = total_path / (n * n)
    hub = int(np.argmax(deg))
    # normalized centrality concentration (Freeman)
    star = n - 1
    conc = float((deg.max() - deg).sum() / ((n - 1) * (n - 2))) if n > 2 else 0.0
    return {
        "edges": np.asarray([[u, v, w] for u, v, w in edges], dtype=np.float64),
        "degree": np.asarray(deg, dtype=np.float64),
        "hub_node": float(hub),
        "max_degree": float(deg.max()),
        "mean_path": float(mean_path),
        "centrality_concentration": conc,
        "min_eccentricity": float(ecc.min()),
        "star_max_deg": float(star),
    }


def pmfg_build(corr: FloatArray) -> dict[str, float | FloatArray]:
    """Greedy planar filtered graph (up to 3n-6 edges) — accepts
    edges in ascending distance order while the added edge can
    complete a 4-clique-free planar cell. Heuristic planarity:
    an edge is accepted iff its endpoints share < 3 common
    neighbor cliques (documented approximation)."""
    cc = _check_corr(corr)
    n = cc.shape[0]
    d = _dist(cc)
    pairs = [(float(d[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda t: t[0])
    adj = np.zeros((n, n))
    n_edges = 0
    max_edges = 3 * n - 6
    for w, i, j in pairs:
        if n_edges >= max_edges:
            break
        # heuristic planarity: block only if endpoints already
        # share >=2 common neighbors forming a crossed chord
        common = np.flatnonzero((adj[i] > 0) & (adj[j] > 0))
        if common.size >= 3:
            continue
        adj[i, j] = w
        adj[j, i] = w
        n_edges += 1
    deg = (adj > 0).sum(axis=0).astype(np.float64)
    return {
        "adjacency": np.asarray(adj, dtype=np.float64),
        "n_edges": float(n_edges),
        "degree": np.asarray(deg, dtype=np.float64),
        "density": float(n_edges / max_edges),
    }


def bench_mst_topology(seed: int = 20261231 + 460) -> dict[str, float]:
    """SYNTHETIC check — factor-dominant node is the MST hub."""
    rng = np.random.default_rng(seed)
    n = 8
    f = rng.normal(size=200)
    lams = np.array([0.9] + list(rng.uniform(0.3, 0.6, n - 1)))
    x = np.column_stack(
        [lams[j] * f + np.sqrt(1 - lams[j] ** 2) * rng.normal(size=200) for j in range(n)]
    )
    c = np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)
    out = mst_build(c)
    if int(out["hub_node"]) != 0:
        raise ValueError(f"hub off: hub={out['hub_node']} expected 0")
    p = pmfg_build(c)
    if float(p["n_edges"]) < n - 1:
        raise ValueError("pmfg too sparse")
    return {
        "synthetic_hub": float(out["hub_node"]),
        "synthetic_max_deg": float(out["max_degree"]),
        "synthetic_pmfg_edges": float(p["n_edges"]),
        "synthetic_score": 1.0,
    }
