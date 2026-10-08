"""Topological data analysis for sliding-window regime detection.

Pipeline: Takens delay embedding turns a 1-D series into an ``R^d`` point
cloud per window; Vietoris–Rips persistent homology summarizes the cloud as
a persistence diagram; diagrams are vectorized into persistence landscapes;
the L2 distance between consecutive window landscapes is a regime-shift
score (loops appearing/vanishing as the window slides across a regime
boundary, in the spirit of Gidea–Katz for crash detection).

Algorithms (all exact for ``n <= ~60`` points unless noted):

- 0-dim persistence: union-find on edges sorted by length — exact bars are
  the accepted edge weights; one essential class per surviving component is
  never born twice and is reported separately from the finite bars.
- 1-dim persistence: full boundary-matrix reduction over Z2 on
  vertices+edges+triangles, ordered by (filtration, dimension). Boundary
  columns are Python ``int`` bitsets so reduction is a handful of XORs per
  column — clarity over speed, still fast at this scale. ``r_max`` bounds
  the filtration (triangles and edges above it are dropped); with
  ``r_max=None`` the complex is complete and the result is exact.
- Bottleneck distance: ``method="greedy"`` sorts candidate pairings and
  takes each when it is no worse than sending both points to the diagonal —
  a documented *approximate* matching, an upper bound on the true distance.
  ``method="exact"`` (default) finds the smallest threshold at which a
  bipartite matching covers every point that cannot afford the diagonal,
  via increasing-cardinality augmenting paths — exact for this multiset
  formulation (Cohen-Steiner et al. 2007).
- Persistence landscapes (Bubenik 2015): ``lambda_k(t)`` is the k-th
  largest tent height ``min(t-b, d-t)+``; discretized on a shared grid, L2
  distance between landscapes lower-bounds Wasserstein/bottleneck costs.

Honesty: bench numbers are SYNTHETIC correctness checks (planted clouds and
planted regime changes), never market evidence; every key is prefixed
``synthetic_``. Diagrams with infinite (essential) bars are excluded from
landscapes and bottleneck inputs by default — they are flagged
``death=inf`` rather than silently truncated.

Composition: complements ``models/koopman_edmd.py`` (operator-theoretic
regime features) and the ``metrics/`` scoring stack — this module supplies
geometry/topology-based regime features and distances only; it does not
score forecasts, touch receipts, or register in the model registry. Pure
numpy + stdlib (scipy not required), deterministic under an explicit seed.

References:
- Takens, F. (1981). Detecting strange attractors in turbulence. *Lecture
  Notes in Mathematics* 898: 366-381. doi:10.1007/BFb0091924.
- Chazal, F., Michel, B. (2021). An introduction to Topological Data
  Analysis. *Frontiers in Artificial Intelligence* 4.
  arXiv:1710.04019.
- Bubenik, P. (2015). Statistical topological data analysis using
  persistence landscapes. *JMLR* 16(3): 77-102. arXiv:1207.6437.
- Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). Stability of
  persistence diagrams. *Discrete & Computational Geometry* 37(1):
  103-120. doi:10.1007/s00454-006-1276-5.
- Gidea, M., Katz, Y. (2018). Topological data analysis of financial time
  series: landscapes of crashes. *Physica A* 491: 820-834.
  arXiv:1703.04385.
- Zomorodian, A., Carlsson, G. (2005). Computing persistent homology.
  *Discrete & Computational Geometry* 33(2): 249-274.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "Landscape",
    "RegimeScores",
    "bench_tda_persistence",
    "bottleneck_distance",
    "diagram_landscape_l2",
    "h0_persistence",
    "h1_persistence",
    "landscape_l2",
    "landscape_norm",
    "persistence_entropy",
    "persistence_landscape",
    "persistence_summary",
    "takens_embedding",
    "tda_regime_scores",
    "vietoris_rips",
]

_MAX_SIMPLICES = 250_000


@dataclass(frozen=True)
class Landscape:
    """Persistence landscape discretized on a grid.

    ``grid`` has shape ``(n_grid,)``; ``values`` has shape
    ``(n_levels, n_grid)`` where ``values[k]`` is ``lambda_{k+1}``.
    """

    grid: Array
    values: Array


class RegimeScores(NamedTuple):
    """Sliding-window regime-shift scores.

    ``scores[i]`` is the landscape distance between the window starting at
    ``starts[i]`` and the window starting at ``starts[i + 1]``.
    """

    scores: Array
    starts: Array


def _as_series(x: Array, name: str = "x", min_len: int = 2) -> Array:
    s = np.asarray(x, dtype=np.float64)
    if s.ndim != 1 or s.shape[0] < min_len or not np.all(np.isfinite(s)):
        raise ValueError(f"{name} must be a finite 1-D array of length >= {min_len}")
    return s


def _as_points(points: Array, name: str = "points", min_rows: int = 2) -> Array:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 1 or p.shape[0] < min_rows or not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must be a finite 2-D array with >= {min_rows} rows")
    return p


def _as_diagram(dgm: Array, name: str = "dgm") -> Array:
    d = np.asarray(dgm, dtype=np.float64)
    if d.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError(f"{name} must have shape (k, 2)")
    if not np.all(np.isfinite(d)):
        raise ValueError(f"{name} must be finite (exclude essential bars first)")
    if np.any(d[:, 1] < d[:, 0]):
        raise ValueError(f"{name}: every death must be >= its birth")
    return d


def _pairwise_dists(points: Array) -> Array:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def takens_embedding(x: Array, dim: int, delay: int = 1) -> Array:
    """Takens (1981) delay embedding of a 1-D series into ``R^dim``.

    Row ``i`` of the output is ``(x[i], x[i+delay], ..., x[i+(dim-1)delay])``;
    the cloud has ``n - (dim-1)*delay`` points. Requires ``dim >= 2``,
    ``delay >= 1`` and at least two embedded points — a single point has no
    informative Vietoris–Rips complex.
    """
    s = _as_series(x, "x", min_len=2)
    if int(dim) < 2:
        raise ValueError("dim must be >= 2")
    if int(delay) < 1:
        raise ValueError("delay must be >= 1")
    n_out = s.shape[0] - (int(dim) - 1) * int(delay)
    if n_out < 2:
        raise ValueError("series too short for the requested (dim, delay)")
    idx = np.arange(n_out)[:, None] + int(delay) * np.arange(int(dim))[None, :]
    return s[idx]


def h0_persistence(points: Array, r_max: float | None = None) -> Array:
    """Exact 0-dim Vietoris–Rips persistence via union-find on sorted edges.

    Every finite H0 bar is born at 0 and dies when an edge merges two
    components, so the bars are exactly the weights of the edges accepted by
    Kruskal's scan (one per merge). ``r_max`` drops edges above the bound,
    which can leave multiple essential components; the finite bars returned
    are still exact for the bounded complex. Essential classes (one per
    surviving component) are topological noise for landscapes/distances and
    are not returned.
    """
    p = _as_points(points)
    n = p.shape[0]
    if r_max is not None and (not np.isfinite(r_max) or r_max < 0.0):
        raise ValueError("r_max must be a non-negative finite value or None")
    dist = _pairwise_dists(p)
    edges: list[tuple[float, int, int]] = []
    for i, j in combinations(range(n), 2):
        w = float(dist[i, j])
        if r_max is None or w <= r_max:
            edges.append((w, i, j))
    edges.sort()

    parent = list(range(n))
    rank = [0] * n

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bars: list[float] = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1
        bars.append(w)
    if not bars:
        return np.empty((0, 2), dtype=np.float64)
    dgm = np.empty((len(bars), 2), dtype=np.float64)
    dgm[:, 0] = 0.0
    dgm[:, 1] = np.asarray(bars)
    return dgm


def _vr_pairs(
    dist: Array, r_max: float | None
) -> tuple[list[tuple[int, int]], list[int], list[float], list[int]]:
    """Boundary-matrix reduction over Z2 on vertices + edges + triangles.

    Returns ``(pairs, creators, fvals, dims)`` in filtration order:
    ``pairs`` are ``(birth_pos, death_pos)`` of sorted simplex positions;
    ``creators`` are sorted positions of positive simplices (columns that
    reduced to zero); ``fvals``/``dims`` are each simplex's filtration value
    and dimension at its sorted position.
    """
    n = dist.shape[0]
    # Fail before materializing anything when the full complex provably
    # exceeds the cap (r_max=None keeps every edge and triangle).
    if r_max is None and n + n * (n - 1) // 2 + n * (n - 1) * (n - 2) // 6 > _MAX_SIMPLICES:
        raise ValueError(
            f"complex has {n + n * (n - 1) // 2 + n * (n - 1) * (n - 2) // 6} simplices "
            f"(cap {_MAX_SIMPLICES}); subsample the cloud or pass a smaller r_max"
        )
    edge_index: dict[tuple[int, int], int] = {}
    fvals: list[float] = [0.0] * n
    dims: list[int] = [0] * n
    edge_verts: list[tuple[int, int]] = []
    for i, j in combinations(range(n), 2):
        w = float(dist[i, j])
        if r_max is not None and w > r_max:
            continue
        edge_index[(i, j)] = len(fvals)
        edge_verts.append((i, j))
        fvals.append(w)
        dims.append(1)
    if len(fvals) > _MAX_SIMPLICES:
        raise ValueError(
            f"complex has {len(fvals)} simplices (cap {_MAX_SIMPLICES}); "
            "subsample the cloud or pass a smaller r_max"
        )
    tri_edges: list[tuple[int, int, int]] = []
    for i, j, k in combinations(range(n), 3):
        e1 = edge_index.get((i, j))
        e2 = edge_index.get((i, k))
        e3 = edge_index.get((j, k))
        if e1 is None or e2 is None or e3 is None:
            continue
        tri_edges.append((e1, e2, e3))
        fvals.append(max(fvals[e1], fvals[e2], fvals[e3]))
        dims.append(2)
        # check incrementally: never materialize the whole C(n,3) complex
        if len(fvals) > _MAX_SIMPLICES:
            raise ValueError(
                f"complex exceeds {_MAX_SIMPLICES} simplices; "
                "subsample the cloud or pass a smaller r_max"
            )
    n_total = len(fvals)

    # Filtration order: (value, dimension, index) — faces before cofaces.
    order = sorted(range(n_total), key=lambda s: (fvals[s], dims[s], s))
    pos_of = [0] * n_total
    f_sorted: list[float] = []
    dims_sorted: list[int] = []
    for pos, s in enumerate(order):
        pos_of[s] = pos
        f_sorted.append(fvals[s])
        dims_sorted.append(dims[s])

    # Boundary of each simplex as a Python int bitset over sorted positions.
    boundary: list[int] = [0] * n_total
    for e_idx, (i, j) in enumerate(edge_verts):
        boundary[n + e_idx] = (1 << pos_of[i]) | (1 << pos_of[j])
    n_edges = len(edge_verts)
    for t_idx, (e1, e2, e3) in enumerate(tri_edges):
        boundary[n + n_edges + t_idx] = (1 << pos_of[e1]) | (1 << pos_of[e2]) | (1 << pos_of[e3])

    low: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    creators: list[int] = []
    for j in range(n_total):
        col = boundary[order[j]]
        while col:
            piv = col.bit_length() - 1
            stored = low.get(piv)
            if stored is None:
                break
            col ^= stored
        if col:
            piv = col.bit_length() - 1
            low[piv] = col
            pairs.append((piv, j))
        else:
            creators.append(j)
    return pairs, creators, f_sorted, dims_sorted


def h1_persistence(
    points: Array, r_max: float | None = None, include_essential: bool = False
) -> Array:
    """1-dim Vietoris–Rips persistence via boundary-matrix reduction.

    A bar ``(birth, death)`` records a loop born when edge ``birth`` closed a
    cycle and killed when the triangle at ``death`` filled it. Essential
    classes (loops never filled within the bounded complex) carry
    ``death=inf`` and are returned only with ``include_essential=True`` —
    they are excluded from landscapes and distances by ``_as_diagram``.
    """
    p = _as_points(points)
    if r_max is not None and (not np.isfinite(r_max) or r_max < 0.0):
        raise ValueError("r_max must be a non-negative finite value or None")
    dist = _pairwise_dists(p)
    pairs, creators, f_sorted, dims_sorted = _vr_pairs(dist, r_max)
    bars: list[tuple[float, float]] = []
    for piv, j in pairs:
        if dims_sorted[piv] == 1 and dims_sorted[j] == 2 and f_sorted[j] > f_sorted[piv]:
            bars.append((f_sorted[piv], f_sorted[j]))
    if include_essential:
        paired = {piv for piv, _ in pairs}
        for c in creators:
            if dims_sorted[c] == 1 and c not in paired:
                bars.append((f_sorted[c], math.inf))
    if not bars:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(bars, dtype=np.float64)


def vietoris_rips(
    points: Array, r_max: float | None = None, include_essential: bool = False
) -> dict[str, Array]:
    """0- and 1-dim Vietoris–Rips persistence diagrams of a point cloud."""
    return {
        "h0": h0_persistence(points, r_max=r_max),
        "h1": h1_persistence(points, r_max=r_max, include_essential=include_essential),
    }


def persistence_landscape(
    dgm: Array,
    n_levels: int = 3,
    n_grid: int = 200,
    t_min: float | None = None,
    t_max: float | None = None,
) -> Landscape:
    """Persistence landscape (Bubenik 2015) discretized on a uniform grid.

    For a diagram point ``(b, d)`` the tent function is
    ``T(t) = max(0, min(t - b, d - t))``; ``lambda_k(t)`` is the k-th
    largest tent height at ``t``. Grid defaults to ``[min birth, max
    death]``; pass shared ``t_min``/``t_max`` for comparable landscapes.
    """
    d = _as_diagram(dgm)
    if int(n_levels) < 1:
        raise ValueError("n_levels must be >= 1")
    if int(n_grid) < 2:
        raise ValueError("n_grid must be >= 2")
    if d.shape[0] == 0:
        lo = 0.0 if t_min is None else float(t_min)
        hi = 1.0 if t_max is None else float(t_max)
        grid = np.linspace(lo, hi, int(n_grid))
        return Landscape(grid=grid, values=np.zeros((int(n_levels), int(n_grid))))
    lo = float(d[:, 0].min()) if t_min is None else float(t_min)
    hi = float(d[:, 1].max()) if t_max is None else float(t_max)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError("landscape grid bounds must be finite with t_max > t_min")
    grid = np.linspace(lo, hi, int(n_grid))
    b = d[:, 0][:, None]
    dd = d[:, 1][:, None]
    tents = np.minimum(grid[None, :] - b, dd - grid[None, :])
    np.maximum(tents, 0.0, out=tents)
    tents.sort(axis=0)
    tents = tents[::-1]
    values = np.zeros((int(n_levels), int(n_grid)))
    m = min(int(n_levels), d.shape[0])
    values[:m] = tents[:m]
    return Landscape(grid=grid, values=values)


def landscape_norm(land: Landscape, p: float = 2.0) -> float:
    """Lp norm of a landscape: ``(sum_k int |lambda_k|^p dt)^(1/p)``."""
    if p == math.inf:
        return float(np.abs(land.values).max())
    if not (np.isfinite(p) and p >= 1.0):
        raise ValueError("p must be finite and >= 1")
    dx = float(land.grid[1] - land.grid[0])
    return float((dx * np.sum(np.abs(land.values) ** p)) ** (1.0 / p))


def _check_same_grid(a: Landscape, b: Landscape) -> None:
    if a.grid.shape != b.grid.shape or not np.allclose(a.grid, b.grid):
        raise ValueError("landscapes must share the same grid")


def landscape_l2(a: Landscape, b: Landscape) -> float:
    """L2 distance between two landscapes on a shared grid."""
    _check_same_grid(a, b)
    dx = float(a.grid[1] - a.grid[0])
    return float(math.sqrt(dx * float(np.sum((a.values - b.values) ** 2))))


def diagram_landscape_l2(dgm_a: Array, dgm_b: Array, n_levels: int = 3, n_grid: int = 200) -> float:
    """L2 landscape distance between two diagrams on their union grid."""
    a = _as_diagram(dgm_a, "dgm_a")
    b = _as_diagram(dgm_b, "dgm_b")
    lo = min(float(a[:, 0].min()) if a.size else 0.0, float(b[:, 0].min()) if b.size else 0.0)
    hi = max(float(a[:, 1].max()) if a.size else 1.0, float(b[:, 1].max()) if b.size else 1.0)
    if hi <= lo:
        hi = lo + 1.0
    la = persistence_landscape(a, n_levels=n_levels, n_grid=n_grid, t_min=lo, t_max=hi)
    lb = persistence_landscape(b, n_levels=n_levels, n_grid=n_grid, t_min=lo, t_max=hi)
    return landscape_l2(la, lb)


def _edge_costs(a: Array, b: Array) -> tuple[Array, Array, Array]:
    """Point-to-point L-infinity costs and per-point diagonal costs."""
    cost = np.abs(a[:, None, :] - b[None, :, :]).max(axis=-1)
    diag_a = (a[:, 1] - a[:, 0]) / 2.0
    diag_b = (b[:, 1] - b[:, 0]) / 2.0
    return cost, diag_a, diag_b


def _bottleneck_greedy(cost: Array, diag_a: Array, diag_b: Array) -> float:
    """Approximate bottleneck: greedy pairing, never worse than both-diagonal.

    Candidates are sorted by cost and taken when both endpoints are free and
    pairing is at most the cost of sending both to the diagonal; leftovers
    go to the diagonal. Documented heuristic — an upper bound on the true
    bottleneck distance, not exact.
    """
    na, nb = cost.shape
    cand = sorted((float(cost[i, j]), i, j) for i in range(na) for j in range(nb))
    used_a: set[int] = set()
    used_b: set[int] = set()
    worst = 0.0
    for c, i, j in cand:
        if i in used_a or j in used_b:
            continue
        if c <= max(diag_a[i], diag_b[j]):
            used_a.add(i)
            used_b.add(j)
            worst = max(worst, c)
    for i in range(na):
        if i not in used_a:
            worst = max(worst, float(diag_a[i]))
    for j in range(nb):
        if j not in used_b:
            worst = max(worst, float(diag_b[j]))
    return worst


def _max_matching(adj: list[list[int]], n_right: int) -> list[int]:
    """Augmenting-path bipartite matching; returns match of right vertices.

    Iterative Kuhn search (explicit stack) — augmenting paths of length
    ~2n blow the recursion limit on large diagrams and must fail into a
    correct answer, not a RecursionError.
    """
    match_r = [-1] * n_right
    for u0 in range(len(adj)):
        seen = [False] * n_right
        # via[u] = right vertex whose matching edge led to left vertex u;
        # disc[v] = left vertex that discovered right vertex v.
        via: dict[int, int] = {}
        disc: dict[int, int] = {}
        stack = [u0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if seen[v]:
                    continue
                seen[v] = True
                disc[v] = u
                w = match_r[v]
                if w < 0:
                    # free right vertex: flip the alternating path back to u0
                    match_r[v] = u
                    while u != u0:
                        v_prev = via[u]
                        u_prev = disc[v_prev]
                        match_r[v_prev] = u_prev
                        u = u_prev
                    break
                via[w] = v
                stack.append(w)
            else:
                continue
            break
    return match_r


def _bottleneck_exact(cost: Array, diag_a: Array, diag_b: Array) -> float:
    """Exact bottleneck distance via threshold + increasing-cardinality matching.

    A point may either be matched across (cost ``c_ij``) or sent to the
    diagonal (cost ``pers/2``). At threshold ``t``, every point whose
    diagonal cost exceeds ``t`` must be cross-matched; the minimum feasible
    ``t`` — which is always one of the candidate edge/diagonal costs — is
    the bottleneck distance.
    """
    na, nb = cost.shape
    candidates = np.unique(np.concatenate([cost.ravel(), diag_a, diag_b, [0.0]]))
    lo_i, hi_i = 0, candidates.shape[0] - 1
    while lo_i < hi_i:
        mid = (lo_i + hi_i) // 2
        t = float(candidates[mid])
        adj: list[list[int]] = [[] for _ in range(na)]
        for i in range(na):
            for j in range(nb):
                if cost[i, j] <= t:
                    adj[i].append(j)
        match_r = _max_matching(adj, nb)
        match_l = [-1] * na
        for v, u in enumerate(match_r):
            if u >= 0:
                match_l[u] = v
        feasible = all(match_l[i] >= 0 for i in range(na) if diag_a[i] > t) and all(
            match_r[j] >= 0 for j in range(nb) if diag_b[j] > t
        )
        if feasible:
            hi_i = mid
        else:
            lo_i = mid + 1
    return float(candidates[lo_i])


def bottleneck_distance(dgm_a: Array, dgm_b: Array, method: str = "exact") -> float:
    """Bottleneck distance between two persistence diagrams.

    ``method="exact"`` (default): threshold-based bipartite matching —
    exact for finite diagrams. ``method="greedy"``: increasing-cardinality
    greedy matching — a documented approximation (upper bound).
    """
    a = _as_diagram(dgm_a, "dgm_a")
    b = _as_diagram(dgm_b, "dgm_b")
    if a.shape[0] == 0 and b.shape[0] == 0:
        return 0.0
    if a.shape[0] == 0:
        return float((b[:, 1] - b[:, 0]).max() / 2.0)
    if b.shape[0] == 0:
        return float((a[:, 1] - a[:, 0]).max() / 2.0)
    cost, diag_a, diag_b = _edge_costs(a, b)
    if method == "exact":
        return _bottleneck_exact(cost, diag_a, diag_b)
    if method == "greedy":
        return _bottleneck_greedy(cost, diag_a, diag_b)
    raise ValueError(f"method must be 'exact' or 'greedy', got {method!r}")


def persistence_entropy(dgm: Array) -> float:
    """Persistence entropy ``-sum p_i log p_i`` over normalized lifetimes."""
    d = _as_diagram(dgm)
    if d.shape[0] == 0:
        return 0.0
    life = d[:, 1] - d[:, 0]
    life = life[life > 0.0]
    if life.shape[0] == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log(p)).sum())


def persistence_summary(dgm: Array, n_levels: int = 3, n_grid: int = 200) -> dict[str, float]:
    """Summary statistics of a diagram: counts, persistence, entropy, norms."""
    d = _as_diagram(dgm)
    life = d[:, 1] - d[:, 0] if d.shape[0] else np.zeros(0)
    land = persistence_landscape(d, n_levels=n_levels, n_grid=n_grid)
    return {
        "n_points": float(d.shape[0]),
        "total_persistence": float(life.sum()),
        "max_persistence": float(life.max()) if life.size else 0.0,
        "mean_persistence": float(life.mean()) if life.size else 0.0,
        "persistence_entropy": persistence_entropy(d),
        "landscape_l1": landscape_norm(land, p=1.0),
        "landscape_l2": landscape_norm(land, p=2.0),
        "landscape_linf": landscape_norm(land, p=math.inf),
    }


def tda_regime_scores(
    x: Array,
    window: int,
    emb_dim: int,
    delay: int = 1,
    stride: int = 1,
    dims: tuple[int, ...] = (0, 1),
    n_levels: int = 3,
    n_grid: int = 64,
    r_max: float | None = None,
    seed: int = 0,
    max_points: int = 60,
    n_ref: int = 0,
) -> RegimeScores:
    """Sliding-window regime-shift scores via landscape L2 distances.

    Each window of ``x`` is Takens-embedded, its persistence diagram(s)
    computed for the requested ``dims`` (0 = components, 1 = loops), and
    vectorized on one shared grid. ``scores[i]`` is attributed to the
    window at ``starts[i]``:

    - ``n_ref >= 1`` (recommended for regime detection): score is the L2
      distance to the *reference landscape* — the pointwise median of the
      first ``n_ref`` windows' landscapes. Within-regime windows score near
      the within-class jitter while post-change windows score high even
      when the topological transition is gradual; this is the
      early-warning formulation of Gidea–Katz (landscape deviations from a
      pre-crash reference), unlike consecutive distances which are
      swamped by within-regime jitter when the morph is slow.
    - ``n_ref = 0`` (consecutive mode): ``scores[i]`` is the combined L2
      distance between windows ``i`` and ``i+1``; ``starts`` drops the
      first window so ``len(scores) == len(starts)``.

    Clouds are capped at ``max_points`` by seeded subsampling — beyond that
    the triangle count makes the bounded reduction unnecessarily heavy.
    """
    s = _as_series(x, "x", min_len=int(window))
    if int(window) < 4:
        raise ValueError("window must be >= 4")
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    if int(max_points) < 2:
        raise ValueError("max_points must be >= 2")
    if int(n_ref) < 0 or int(n_ref) >= int((s.shape[0] - int(window)) // int(stride) + 1):
        raise ValueError("n_ref must be in [0, n_windows)")
    if not dims or any(d not in (0, 1) for d in dims):
        raise ValueError("dims must be a non-empty tuple drawn from {0, 1}")
    starts = np.arange(0, s.shape[0] - int(window) + 1, int(stride))
    if starts.shape[0] < 2:
        raise ValueError("series too short: fewer than two windows")
    rng = np.random.default_rng(seed)

    dgms: list[dict[int, Array]] = []
    t_hi = 1e-12
    for st in starts:
        pts = takens_embedding(s[st : st + int(window)], int(emb_dim), int(delay))
        if pts.shape[0] > int(max_points):
            pts = pts[rng.choice(pts.shape[0], int(max_points), replace=False)]
        per_dim: dict[int, Array] = {}
        for d in dims:
            if d == 0:
                dg = h0_persistence(pts, r_max=r_max)
            else:
                dg = h1_persistence(pts, r_max=r_max)
            per_dim[d] = dg
            if dg.shape[0]:
                t_hi = max(t_hi, float(dg[:, 1].max()))
        dgms.append(per_dim)

    lands: list[dict[int, Landscape]] = []
    for per_dim in dgms:
        lands.append(
            {
                d: persistence_landscape(
                    dg, n_levels=n_levels, n_grid=n_grid, t_min=0.0, t_max=t_hi
                )
                for d, dg in per_dim.items()
            }
        )
    n_win = len(lands)
    if int(n_ref) >= 1:
        ref_vals = {
            d: np.median(np.stack([lands[i][d].values for i in range(int(n_ref))]), axis=0)
            for d in dims
        }
        scores = np.empty(n_win)
        for i in range(n_win):
            acc = 0.0
            for d in dims:
                diff = lands[i][d].values - ref_vals[d]
                acc += float(np.sum(diff * diff))
            scores[i] = math.sqrt(acc)
        return RegimeScores(scores=scores, starts=starts.astype(np.float64))
    scores = np.empty(n_win - 1)
    for i in range(n_win - 1):
        acc = 0.0
        for d in dims:
            diff = lands[i][d].values - lands[i + 1][d].values
            acc += float(np.sum(diff * diff))
        scores[i] = math.sqrt(acc)
    return RegimeScores(scores=scores, starts=starts[1:].astype(np.float64))


def _rank_auc(scores: Array, labels: Array) -> float:
    """Mann–Whitney AUC with tie handling; NaN when a class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.shape[0])
    ranks[order] = np.arange(1, s.shape[0] + 1)
    # Average ranks for ties.
    for v in np.unique(s):
        mask = s == v
        if int(mask.sum()) > 1:
            ranks[mask] = float(ranks[mask].mean())
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def bench_tda_persistence(seed: int = 7) -> dict[str, float]:
    """SYNTHETIC correctness bench — planted numbers only, never market evidence.

    Returns floats (all keys ``synthetic_`` prefixed):
      - ``synthetic_landscape_sep``: cross-class minus within-class mean
        landscape L2 distance (circle-embedded vs i.i.d. noise windows).
      - ``synthetic_h1_circle_over_noise``: ratio of the most persistent H1
        bar on a noisy-circle cloud vs a Gaussian cloud.
      - ``synthetic_bottleneck_self``: d_B(D, D) — must be ~0.
      - ``synthetic_bottleneck_err``: |d_B - delta| on a diagram shifted by
        a planted ``delta`` (matching stays point-to-point by construction).
      - ``synthetic_regime_auc``: AUC of the landscape score against windows
        straddling a planted AR(1)-to-periodic regime change.
      - ``synthetic_entropy_noise_minus_circle``: entropy contrast.
    """
    rng = np.random.default_rng(int(seed))
    out: dict[str, float] = {}

    # --- Landscape separation: noisy-circle cloud vs Gaussian cloud -------
    t = np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False)
    win = 10
    circle_dgms = []
    noise_dgms = []
    for _w in range(win):
        th = t + 0.03 * rng.standard_normal(t.shape[0])
        cloud_c = np.column_stack([np.cos(th), np.sin(th)])
        cloud_c += 0.05 * rng.standard_normal(cloud_c.shape)
        cloud_n = rng.standard_normal((28, 2))
        circle_dgms.append(h1_persistence(cloud_c))
        noise_dgms.append(h1_persistence(cloud_n))
    all_dgms = circle_dgms + noise_dgms
    hi = max(float(d[:, 1].max()) if d.shape[0] else 0.0 for d in all_dgms)
    lands = [
        persistence_landscape(d, n_levels=3, n_grid=64, t_min=0.0, t_max=max(hi, 1e-6))
        for d in all_dgms
    ]

    def _mean_pairwise(idxs_a: list[int], idxs_b: list[int]) -> float:
        vals = [landscape_l2(lands[i], lands[j]) for i in idxs_a for j in idxs_b]
        return float(np.mean(vals))

    idx_c = list(range(win))
    idx_n = list(range(win, 2 * win))
    within = 0.5 * (_mean_pairwise(idx_c, idx_c) + _mean_pairwise(idx_n, idx_n))
    cross = _mean_pairwise(idx_c, idx_n)
    out["synthetic_landscape_sep"] = cross - within
    out["synthetic_landscape_sep_ratio"] = cross / within if within > 0 else math.inf

    # --- H1 contrast ------------------------------------------------------
    circle_top = max(
        float(d[:, 1].max() - d[:, 0].min()) if d.shape[0] else 0.0 for d in circle_dgms
    )
    noise_top = max(float(d[:, 1].max() - d[:, 0].min()) if d.shape[0] else 0.0 for d in noise_dgms)
    out["synthetic_h1_circle_over_noise"] = circle_top / noise_top if noise_top > 0 else math.inf
    ent_c = float(np.mean([persistence_entropy(d) for d in circle_dgms]))
    ent_n = float(np.mean([persistence_entropy(d) for d in noise_dgms]))
    out["synthetic_entropy_noise_minus_circle"] = ent_n - ent_c

    # --- Bottleneck sanity -------------------------------------------------
    n_pts = 8
    births = rng.uniform(0.0, 0.4, n_pts)
    deaths = births + rng.uniform(2.0, 4.0, n_pts)
    dgm_a = np.column_stack([births, deaths])
    delta = 0.3
    dgm_b = dgm_a + delta
    out["synthetic_bottleneck_self"] = bottleneck_distance(dgm_a, dgm_a)
    out["synthetic_bottleneck_err"] = abs(bottleneck_distance(dgm_a, dgm_b) - delta)
    # Greedy matching is approximate: the gap over exact is >= 0.
    out["synthetic_bottleneck_greedy_minus_exact"] = bottleneck_distance(
        dgm_a, dgm_b, method="greedy"
    ) - bottleneck_distance(dgm_a, dgm_b)

    # --- Regime-detection AUC ---------------------------------------------
    # Planted topological regime change: i.i.d. blob windows -> clean orbit
    # windows (delay ~ period/4 gives a fat elliptic cloud, so a real loop
    # appears). Scored against the median reference landscape of the first
    # n_ref windows; positive = window fully inside the orbit regime.
    n1, n2 = 240, 240
    e = rng.standard_normal(n1 + n2)
    x_iid = 0.5 * e[:n1]
    tt = np.arange(n2)
    x_per = 1.5 * np.sin(2.0 * math.pi * tt / 24.0) + 0.15 * e[n1:]
    series = np.concatenate([x_iid, x_per])
    res = tda_regime_scores(
        series,
        window=32,
        emb_dim=2,
        delay=4,
        stride=4,
        dims=(1,),
        n_grid=48,
        n_ref=15,
    )
    boundary = float(n1)
    labels = (res.starts >= boundary).astype(np.float64)
    out["synthetic_regime_auc"] = _rank_auc(res.scores, labels)
    pos_scores = res.scores[labels > 0]
    neg_scores = res.scores[labels <= 0]
    out["synthetic_regime_pos_minus_neg"] = (
        float(pos_scores.mean() - neg_scores.mean())
        if pos_scores.size and neg_scores.size
        else math.nan
    )
    out["synthetic_seed"] = float(seed)
    return {k: float(v) for k, v in out.items()}
