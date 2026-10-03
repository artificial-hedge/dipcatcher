"""General regular-vine (R-vine) engine.

Implements the four pieces the C/D-vine special cases cannot express:

* **Structure selection** — Dißmann, Brechmann, Czado & Kurowicka (2013)
  sequential maximum-spanning-tree selection over |Kendall tau|.
* **Peeling** — vinecopulib ``RVineTrees::peel`` (trees → natural-order
  structure array + diagonal order), which converts an arbitrary regular
  tree sequence into the matrix layout the array algorithms consume.
* **Evaluation** — edge-DAG replay (each pair-copula consumes the
  Rosenblatt transforms produced by its parent edges), equivalent to
  VineCopula ``dissmann_loglik`` / vinecopulib ``RVineLogLik``.
* **Sampling** — VineCopula ``rvine_sim``: the inverse-Rosenblatt
  ``vdirect``/``vindirect`` recursion driven by ``MaxMat``/``CondDistr``
  lookup tables built from the vine matrix.

Conventions
-----------
Variable labels are 1..d throughout the vine-matrix machinery (the R /
VineCopula convention); caller columns are 0-based and mapped at the
boundary.  ``pair_copula`` families are the ``_FAMILY_PALETTE`` names from
:mod:`quant_fund.models.pair_vine_copula`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sstats
from scipy.special import gammaln

from quant_fund.models.pair_vine_copula import (
    _FAMILY_PALETTE,
    Array,
    _edge_loglik,
    _h_eval,
    _hinv_eval,
    _kendall_tau_pair,
    _select_family,
)

_EPS = 1e-12


# ------------------------------------------------------------------ types


@dataclass(frozen=True)
class RVineEdge:
    """One fitted pair-copula edge ``(x, y | cond)`` of a regular vine.

    ``left_node``/``right_node`` are the ids of the parent nodes whose
    conditional arrays this edge consumes (variable ids ``0..d-1`` at
    tree 0, otherwise the ``new_node`` ids assigned during selection).
    ``new_node`` is this edge's own node id in the next tree.
    """

    tree: int
    edge_id: int
    left_node: int
    right_node: int
    new_node: int
    x: int
    y: int
    cond: tuple[int, ...]
    family: str
    params: dict[str, float]
    loglik: float


@dataclass
class _Node:
    node_id: int
    complete: frozenset[int]
    cond_data: dict[tuple[int, frozenset[int]], Array]
    parent_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass
class RVineSpec:
    """A fully specified R-vine: tree edges plus peeled matrix form."""

    dim: int
    edges: list[list[RVineEdge]]
    order: np.ndarray  # order[j] = var label (1..d) at vine position j
    struct: np.ndarray  # S[t][e]: partner var label for tree t, column e
    matrix: np.ndarray  # lower-triangular vine matrix, labels 1..d
    maxmat: np.ndarray
    direct: np.ndarray
    indirect: np.ndarray
    fam_cell: np.ndarray  # object array of family names per matrix cell
    par_cell: np.ndarray  # object array of param dicts per matrix cell
    loglik: float = float("nan")


# -------------------------------------------------- structure selection


def _candidate_edges(nodes: list[_Node], tree: int) -> list[tuple]:
    candidates: list[tuple] = []
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            left, right = nodes[i], nodes[j]
            shared: frozenset[int]
            if tree == 1:
                shared = frozenset()
            else:
                if not (left.parent_ids & right.parent_ids):
                    continue
                shared = left.complete & right.complete
                if len(shared) != tree - 1:
                    continue
            ld = tuple(left.complete - shared)
            rd = tuple(right.complete - shared)
            if len(ld) != 1 or len(rd) != 1:
                continue
            x_var, y_var = ld[0], rd[0]
            x = left.cond_data.get((x_var, shared))
            y = right.cond_data.get((y_var, shared))
            if x is None or y is None:
                continue
            tau = _kendall_tau_pair(x, y)
            if not np.isfinite(tau):
                tau = 0.0
            candidates.append((abs(tau), tau, i, j, x_var, y_var, shared, x, y))
    return candidates


def _mst_select(candidates: list[tuple], n_nodes: int) -> list[tuple]:
    """Greedy maximum spanning tree on |tau| weights (union-find)."""
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    selected = []
    for cand in sorted(candidates, key=lambda c: c[0], reverse=True):
        if union(cand[2], cand[3]):
            selected.append(cand)
            if len(selected) == n_nodes - 1:
                break
    if len(selected) != n_nodes - 1:
        raise ValueError("R-vine candidate graph is disconnected")
    return selected


def rvine_fit(
    u: Array,
    families: Sequence[str] = _FAMILY_PALETTE,
    criterion: str = "aic",
    tau_threshold: float = 0.0,
    n_trees: int | None = None,
) -> RVineSpec:
    """Sequential Dißmann (2013) structure selection on pseudo-observations.

    ``u`` is (n, d) with values in (0, 1) — callers should apply their own
    probability-integral transform.  Variables keep their caller column
    indices; no permutation is applied to ``u`` (the ordering emerges in
    ``spec.order``).
    """
    arr = np.asarray(u, dtype=float)
    if arr.ndim != 2:
        raise ValueError("u must be a two-dimensional matrix")
    keep = np.all(np.isfinite(arr), axis=1)
    arr = arr[keep]
    if arr.shape[0] < 20:
        raise ValueError("not enough complete observations for vine selection")
    arr = np.clip(arr, _EPS, 1.0 - _EPS)
    d = int(arr.shape[1])
    if d < 2:
        raise ValueError("vine needs at least two columns")
    max_tree = d - 1 if n_trees is None else min(int(n_trees), d - 1)
    if max_tree < 1:
        raise ValueError("n_trees must be at least 1")

    node_id = 0
    current: list[_Node] = []
    for var in range(d):
        current.append(
            _Node(
                node_id=node_id,
                complete=frozenset({var}),
                cond_data={(var, frozenset()): arr[:, var].copy()},
            )
        )
        node_id += 1

    all_edges: list[list[RVineEdge]] = []
    total_ll = 0.0
    edge_id = 0
    for tree in range(1, max_tree + 1):
        candidates = _candidate_edges(current, tree)
        selected = _mst_select(candidates, len(current))
        nxt: list[_Node] = []
        tree_edges: list[RVineEdge] = []
        for _w, tau, li, ri, x_var, y_var, shared, x, y in selected:
            left, right = current[li], current[ri]
            if abs(tau) <= tau_threshold:
                fam = "gaussian"
                par = {"rho": 0.0, "loglik": 0.0, "aic": 0.0, "bic": 0.0}
            else:
                fam, par, _ = _select_family(x, y, families=tuple(families), criterion=criterion)
            ll = _edge_loglik(fam, par, x, y)
            total_ll += ll
            # left_given = h(x|y) = ∂C/∂y ; right_given = h(y|x) = ∂C/∂x
            left_given = np.clip(_h_eval(x, y, fam, par), _EPS, 1.0 - _EPS)
            right_given = np.clip(_h_eval(y, x, fam, par), _EPS, 1.0 - _EPS)
            left_cond = frozenset(set(shared) | {y_var})
            right_cond = frozenset(set(shared) | {x_var})
            new_node = _Node(
                node_id=node_id,
                complete=left.complete | right.complete,
                cond_data={(x_var, left_cond): left_given, (y_var, right_cond): right_given},
                parent_ids=frozenset({left.node_id, right.node_id}),
            )
            nxt.append(new_node)
            tree_edges.append(
                RVineEdge(
                    tree=tree,
                    edge_id=edge_id,
                    left_node=left.node_id,
                    right_node=right.node_id,
                    new_node=node_id,
                    x=int(x_var),
                    y=int(y_var),
                    cond=tuple(sorted(int(v) for v in shared)),
                    family=fam,
                    params=par,
                    loglik=float(ll),
                )
            )
            node_id += 1
            edge_id += 1
        all_edges.append(tree_edges)
        current = nxt
        if len(current) <= 1:
            break

    order, struct, cell_map = _peel(all_edges, d)
    matrix = _to_vine_matrix(order, struct, d)
    # MaxMat / CondDistr are computed on the *normalized* matrix (diag
    # relabelled d..1), matching the convention rvine_sample consumes.
    matrix_n = _reorder_rvine_matrix(matrix)
    maxmat = _create_max_mat(matrix_n)
    direct, indirect = _needed_cond_distr(matrix_n)

    d2 = d
    fam_cell = np.empty((d2, d2), dtype=object)
    par_cell = np.empty((d2, d2), dtype=object)
    for i in range(d2):
        for j in range(d2):
            fam_cell[i, j] = "gaussian"
            par_cell[i, j] = {"rho": 0.0, "loglik": 0.0, "aic": 0.0, "bic": 0.0}
    # cell (k, i) [k > i] == tree t = d-1-k, vine column i
    for (t, e, _src), edge in cell_map.items():
        k = d - 1 - t
        fam_cell[k, e] = edge.family
        par_cell[k, e] = edge.params

    return RVineSpec(
        dim=d,
        edges=all_edges,
        order=order,
        struct=struct,
        matrix=matrix,
        maxmat=maxmat,
        direct=direct,
        indirect=indirect,
        fam_cell=fam_cell,
        par_cell=par_cell,
        loglik=total_ll,
    )


# ---------------------------------------------------------- peeling


def _peel(
    all_edges: list[list[RVineEdge]], d: int
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int, int], RVineEdge]]:
    """vinecopulib ``RVineTrees::peel`` — trees → (order, S, cell->edge).

    ``order[col]`` is the variable label (1..d) peeled at column ``col``;
    ``struct[t][col]`` is the partner label at tree ``t`` (0-based tree
    index, t < d-1-col).  ``cell_map[(t, col, src)]`` binds the matrix cell
    ``(k=d-1-t, i=col)`` to its source edge.
    """
    trunc = len(all_edges)
    # augmented trees: edges of tree t as a graph on the previous tree's
    # edges (t=0 nodes are the vars themselves, 1-based labels).
    aug_nodes: list[list[tuple[int, int]]] = []  # per tree: (node1, node2)
    aug_edges: list[list[RVineEdge]] = []
    for t, tree_edges in enumerate(all_edges):
        if t == 0:
            aug_nodes.append([(e.x + 1, e.y + 1) for e in tree_edges])
        else:
            # keys: (var, C ∪ {partner}) with 1-based labels — the incident
            # node of the line graph on the previous tree's edges.
            lookup: dict[tuple[int, tuple[int, ...]], int] = {}
            for i, pe in enumerate(all_edges[t - 1]):
                base = {v + 1 for v in pe.cond}
                k1 = (pe.x + 1, tuple(sorted(base | {pe.y + 1})))
                k2 = (pe.y + 1, tuple(sorted(base | {pe.x + 1})))
                lookup[k1] = i
                lookup[k2] = i
            nodes = []
            for e in tree_edges:
                c = tuple(sorted(v + 1 for v in e.cond))
                k1 = (e.x + 1, c)
                k2 = (e.y + 1, c)
                if k1 not in lookup or k2 not in lookup:
                    raise ValueError(f"proximity condition violated peeling tree {t}")
                nodes.append((lookup[k1], lookup[k2]))
            aug_nodes.append(nodes)
        aug_edges.append(tree_edges)

    order = np.zeros(d, dtype=int)
    struct = np.zeros((trunc, d), dtype=int)
    cell_map: dict[tuple[int, int, int], RVineEdge] = {}
    consumed: list[set[int]] = [set() for _ in range(trunc)]

    def degree(t: int, node: int) -> int:
        deg = 0
        for i, (n1, n2) in enumerate(aug_nodes[t]):
            if i in consumed[t]:
                continue
            if n1 == node or n2 == node:
                deg += 1
        return deg

    for col in range(d - 1):
        t = max(min(trunc, d - 1 - col), 1)
        tree_idx = t - 1
        # leaf edges: unconsumed edges with a degree-1 endpoint
        diag = -1
        for i, e in enumerate(aug_edges[tree_idx]):
            if i in consumed[tree_idx]:
                continue
            n1, n2 = aug_nodes[tree_idx][i]
            if degree(tree_idx, n1) == 1:
                diag = e.x + 1
                break
            if degree(tree_idx, n2) == 1:
                diag = e.y + 1
                break
        if diag < 0:
            raise ValueError(f"no leaf found while peeling column {col}")
        order[col] = diag

        check_set: list[int] = []
        found = False
        for i, e in enumerate(aug_edges[tree_idx]):
            if i in consumed[tree_idx]:
                continue
            n1, n2 = aug_nodes[tree_idx][i]
            as_a = degree(tree_idx, n1) == 1 and e.x + 1 == diag
            as_b = degree(tree_idx, n2) == 1 and e.y + 1 == diag
            if not (as_a or as_b):
                continue
            struct[tree_idx, col] = e.y + 1 if as_a else e.x + 1
            cell_map[(tree_idx, col, i)] = e
            check_set = sorted(v + 1 for v in e.cond)
            consumed[tree_idx].add(i)
            found = True
            break
        if not found:
            raise ValueError(f"diagonal variable not a leaf at column {col}")

        for k in range(1, t):
            tidx = tree_idx - k
            check_set = sorted(set(check_set) | {diag})
            matched = False
            for i, e in enumerate(aug_edges[tidx]):
                if i in consumed[tidx]:
                    continue
                all_idx = tuple(sorted({e.x + 1, e.y + 1} | {v + 1 for v in e.cond}))
                if list(all_idx) != check_set:
                    continue
                struct[tidx, col] = e.y + 1 if e.x + 1 == diag else e.x + 1
                cell_map[(tidx, col, i)] = e
                check_set = sorted(v + 1 for v in e.cond)
                consumed[tidx].add(i)
                matched = True
                break
            if not matched:
                raise ValueError(f"proximity condition violated peeling column {col}")

    order[d - 1] = struct[0, d - 2]
    return order, struct, cell_map


def _to_vine_matrix(order: np.ndarray, struct: np.ndarray, d: int) -> np.ndarray:
    """(order, S) natural-order form → lower-triangular vine matrix.

    ``M[i][i] = order[i]``; for ``k > i``, ``M[k][i] = S[d-1-k][i]`` —
    the partner label at tree ``d-1-k`` of vine column ``i``.
    """
    m = np.zeros((d, d), dtype=int)
    for i in range(d):
        m[i, i] = order[i]
    for k in range(1, d):
        t = d - 1 - k
        for i in range(d):
            if k > i and t >= 0 and struct[t, i] != 0:
                m[k, i] = struct[t, i]
    return m


def _reorder_rvine_matrix(matrix: np.ndarray) -> np.ndarray:
    mat = np.array(matrix, dtype=int, copy=True)
    n = mat.shape[0]
    old_order = np.diag(mat).copy()
    mapping = {int(old_order[i]): n - i for i in range(n)}
    out = mat.copy()
    for old, new in mapping.items():
        out[mat == old] = new
    return out


def _create_max_mat(matrix: np.ndarray) -> np.ndarray:
    original = np.asarray(matrix, dtype=int)
    maxmat = _reorder_rvine_matrix(original)
    n = maxmat.shape[0]
    for j in range(n - 1):
        for i in range(n - 2, j - 1, -1):
            maxmat[i, j] = int(np.max(maxmat[i : i + 2, j]))
    tmp = maxmat.copy()
    old_sort = np.diag(original)[::-1]
    for i in range(1, n + 1):
        maxmat[tmp == i] = int(old_sort[i - 1])
    return maxmat


def _needed_cond_distr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vine = _reorder_rvine_matrix(matrix)
    maxmat = _create_max_mat(vine)
    d = vine.shape[0]
    direct = np.zeros((d, d), dtype=bool)
    indirect = np.zeros((d, d), dtype=bool)
    direct[1:d, 0] = True
    for i0 in range(1, d - 1):
        v = d - i0
        bw = maxmat[i0:d, 0:i0] == v
        is_direct = vine[i0:d, 0:i0] == v
        indirect[i0:d, i0] = np.any(bw & (~is_direct), axis=1)
        direct[i0:d, i0] = True
        direct[i0, i0] = bool(np.any(bw[0, :] & is_direct[0, :]))
    return direct, indirect


# ------------------------------------------------------ edge-replay eval


def rvine_logpdf_edges(u: Array, spec: RVineSpec) -> np.ndarray:
    """Per-observation vine log-density via edge-DAG replay.

    Equivalent to ``rvine_loglik`` on the peeled matrix but operates on the
    edge records directly — no ``MaxMat`` convention to decode.  Returns a
    length-n vector (already summed over edges).
    """
    arr = np.asarray(u, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != spec.dim:
        raise ValueError("u must be (n, dim)")
    arr = np.clip(arr, _EPS, 1.0 - _EPS)
    nodes: dict[int, _Node] = {}
    for var in range(spec.dim):
        nodes[var] = _Node(
            node_id=var,
            complete=frozenset({var}),
            cond_data={(var, frozenset()): arr[:, var].copy()},
        )
    pointwise = np.zeros(arr.shape[0], dtype=float)
    for tree in spec.edges:
        for edge in tree:
            left, right = nodes[edge.left_node], nodes[edge.right_node]
            shared = frozenset(edge.cond)
            x = left.cond_data[(edge.x, shared)]
            y = right.cond_data[(edge.y, shared)]
            pointwise += _vec_logpdf(edge.family, edge.params, x, y)
            left_given = np.clip(_h_eval(x, y, edge.family, edge.params), _EPS, 1 - _EPS)
            right_given = np.clip(_h_eval(y, x, edge.family, edge.params), _EPS, 1 - _EPS)
            left_cond = frozenset(set(shared) | {edge.y})
            right_cond = frozenset(set(shared) | {edge.x})
            nodes[edge.new_node] = _Node(
                node_id=edge.new_node,
                complete=left.complete | right.complete,
                cond_data={(edge.x, left_cond): left_given, (edge.y, right_cond): right_given},
                parent_ids=frozenset({left.node_id, right.node_id}),
            )
    return pointwise


def _vec_logpdf(fam: str, par: dict[str, float], x: Array, y: Array) -> Array:
    """Per-observation log pair-copula density.

    Mirrors the scalar ``_LOGLIK_FN`` formulas in pair_vine_copula without
    the ``np.sum`` — needed because the vine's total log-density is the
    element-wise sum over edges, not a sum over observations first.
    """
    fn = _VEC_FN.get(fam)
    if fn is None:
        raise ValueError(f"unknown family {fam}")
    if fam == "t":
        return fn(x, y, float(par["rho"]), float(par["nu"]))
    if fam == "gaussian":
        return fn(x, y, float(par["rho"]))
    if fam == "clayton":
        return fn(x, y, float(par["theta"]))
    if fam == "gumbel":
        return fn(x, y, float(par["alpha"]))
    return fn(x, y, float(par["theta"]))


def _gauss_vec(u: Array, v: Array, rho: float) -> Array:
    x = np.asarray(sstats.norm.ppf(np.clip(u, _EPS, 1.0 - _EPS)), dtype=float)
    y = np.asarray(sstats.norm.ppf(np.clip(v, _EPS, 1.0 - _EPS)), dtype=float)
    r2 = rho * rho
    q = (x * x - 2.0 * rho * x * y + y * y) / (1.0 - r2)
    return np.asarray(-0.5 * math.log(1.0 - r2) - 0.5 * q + 0.5 * (x * x + y * y), dtype=float)


def _t_vec(u: Array, v: Array, rho: float, nu: float) -> Array:
    x = np.asarray(sstats.t.ppf(np.clip(u, _EPS, 1.0 - _EPS), df=nu), dtype=float)
    y = np.asarray(sstats.t.ppf(np.clip(v, _EPS, 1.0 - _EPS), df=nu), dtype=float)
    r2 = rho * rho
    d = (x * x + y * y - 2.0 * rho * x * y) / ((1.0 - r2) * nu)
    joint = (
        gammaln((nu + 2.0) / 2.0)
        - gammaln(nu / 2.0)
        - math.log(nu * math.pi)
        - 0.5 * math.log(1.0 - r2)
    ) - (nu + 2.0) / 2.0 * np.log1p(d)
    marg_const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * math.log(nu * math.pi)
    marg_x = marg_const - (nu + 1.0) / 2.0 * np.log1p(x * x / nu)
    marg_y = marg_const - (nu + 1.0) / 2.0 * np.log1p(y * y / nu)
    return np.asarray(joint - marg_x - marg_y, dtype=float)


def _clayton_vec(u: Array, v: Array, theta: float) -> Array:
    uu = np.clip(u, _EPS, 1.0 - _EPS)
    vv = np.clip(v, _EPS, 1.0 - _EPS)
    t1 = math.log(1.0 + theta)
    t2 = (theta + 1.0) * np.log(uu * vv)
    t3 = (2.0 + 1.0 / theta) * np.log(uu ** (-theta) + vv ** (-theta) - 1.0)
    return np.asarray(np.full(uu.shape, t1, dtype=float) - t2 - t3, dtype=float)


def _gumbel_vec(u: Array, v: Array, alpha: float) -> Array:
    if alpha < 1.0001:
        return np.zeros(np.asarray(u).shape, dtype=float)
    uu = np.clip(u, _EPS, 1.0 - _EPS)
    vv = np.clip(v, _EPS, 1.0 - _EPS)
    a = (-np.log(uu)) ** alpha
    b = (-np.log(vv)) ** alpha
    s = a + b
    cap = s ** (1.0 / alpha)
    return np.asarray(
        -cap
        + (1.0 / alpha - 2.0) * np.log(s)
        + np.log(cap + alpha - 1.0)
        + (alpha - 1.0) * (np.log(np.log(1.0 / uu)) + np.log(np.log(1.0 / vv)))
        + (np.log(1.0 / uu) + np.log(1.0 / vv)),
        dtype=float,
    )


def _frank_vec(u: Array, v: Array, theta: float) -> Array:
    if abs(theta) < 1e-8:
        return np.zeros(np.asarray(u).shape, dtype=float)
    uu = np.clip(u, _EPS, 1.0 - _EPS)
    vv = np.clip(v, _EPS, 1.0 - _EPS)
    a = -np.expm1(-theta)
    bu = -np.expm1(-theta * uu)
    bv = -np.expm1(-theta * vv)
    denom = a - bu * bv
    return np.asarray(
        math.log(abs(theta) * abs(a))
        - theta * (uu + vv)
        - 2.0 * np.log(np.maximum(np.abs(denom), 1e-300)),
        dtype=float,
    )


def _joe_vec(u: Array, v: Array, theta: float) -> Array:
    uu = np.clip(u, _EPS, 1.0 - _EPS)
    vv = np.clip(v, _EPS, 1.0 - _EPS)
    a = (1.0 - uu) ** theta
    b = (1.0 - vv) ** theta
    s = a + b - a * b
    return np.asarray(
        (1.0 / theta - 2.0) * np.log(s)
        + (theta - 1.0) * np.log((1.0 - uu) * (1.0 - vv))
        + np.log(theta - 1.0 + s),
        dtype=float,
    )


_VEC_FN: dict[str, Callable[..., Array]] = {
    "gaussian": _gauss_vec,
    "t": _t_vec,
    "clayton": _clayton_vec,
    "gumbel": _gumbel_vec,
    "frank": _frank_vec,
    "joe": _joe_vec,
}

# -------------------------------------------------------- array sampling


def rvine_sample(n: int, spec: RVineSpec, rng: np.random.Generator) -> np.ndarray:
    """Inverse-Rosenblatt sampling (VineCopula ``rvine_sim`` port).

    Returns an (n, dim) array whose columns are in the caller's original
    variable order (the reverse of the ``order`` permutation applied to
    the draws before peeling).
    """
    d = spec.dim
    # normalized labels: mapping old label -> position label (1..d)
    mat_n = _reorder_rvine_matrix(spec.matrix)
    fam = spec.fam_cell[::-1, ::-1]
    par = spec.par_cell[::-1, ::-1]
    maxmat = spec.maxmat[::-1, ::-1]
    mat = mat_n[::-1, ::-1]
    cind = spec.indirect[::-1, ::-1]
    order = np.diag(spec.matrix)  # original labels per vine position
    u2 = rng.random((n, d))

    out = np.zeros((n, d), dtype=float)
    vdirect = np.zeros((d, d), dtype=float)
    vindirect = np.zeros((d, d), dtype=float)
    for row in range(n):
        for i in range(d):
            vdirect[i, i] = u2[row, i]
        vindirect[0, 0] = vdirect[0, 0]
        for i in range(1, d):
            for k in range(i - 1, -1, -1):
                f = fam[k, i]
                p = par[k, i]
                m = int(maxmat[k, i])
                if int(mat[k, i]) == m:
                    cond = vdirect[k, m - 1]
                else:
                    cond = vindirect[k, m - 1]
                # hinv1(cond, w): y s.t. h(y|cond)=w  -> _hinv_eval(w, cond)
                vdirect[k, i] = _hinv_eval(
                    np.asarray([vdirect[k + 1, i]]), np.asarray([cond]), f, p
                )[0]
                if i + 1 < d and cind[k + 1, i]:
                    # hfunc2(a, b) = h(a|b) -> _h_eval(a, b)
                    if int(mat[k, i]) == m:
                        vindirect[k + 1, i] = _h_eval(
                            np.asarray([vdirect[k, m - 1]]),
                            np.asarray([vdirect[k, i]]),
                            f,
                            p,
                        )[0]
                    else:
                        vindirect[k + 1, i] = _h_eval(
                            np.asarray([vindirect[k, m - 1]]),
                            np.asarray([vdirect[k, i]]),
                            f,
                            p,
                        )[0]
        out[row, :] = vdirect[0, :]
    # peel position i holds var order[i]; restore caller order
    ix = np.argsort(order[::-1])
    return out[:, ix]
