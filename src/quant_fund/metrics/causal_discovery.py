"""Constraint- and score-based causal structure discovery.

Recovers a DAG skeleton + oriented edges (CPDAG) from observational
continuous data assuming linear-Gaussian local mechanisms — the regime in
which conditional-independence tests reduce to partial correlation.

Functions
---------
- :func:`partial_corr` — partial correlation + Fisher-z p-value.
- :func:`pc_skeleton` — stable-PC skeleton phase with separation sets.
- :func:`orient_vstructures` / :func:`apply_meek_rules` — CPDAG orientation.
- :func:`pc_fit` — full PC pipeline → :class:`PCResult`.
- :func:`ges_fit` — greedy forward-equivalence search (Gaussian BIC).
- :func:`lingam_pairwise` — LiNGAM-style direction for two variables via
  third-order cumulant asymmetry (DirectLiNGAM-lite; documented).
- :func:`synth_dag` — seeded linear-Gaussian data on a random DAG.
- :func:`bench_causal_discovery` — SYNTHETIC telemetry blob.

References
----------
- Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and
  Search*, 2nd ed. MIT Press — book.
- Chickering (2002). Optimal structure identification with greedy
  search. *JMLR* 3:507 — journal.
- Colombo & Maathuis (2014). Order-independent constraint-based causal
  structure learning. *JMLR* 15 — arXiv:1211.3295.
- Meek (1995). Causal inference and causal explanation with background
  knowledge. *UAI* 95 — conference.
- Shimizu et al. (2006). A linear non-Gaussian acyclic model for causal
  discovery. *JMLR* 7:2003 — journal.

Honesty
-------
All numbers are SYNTHETIC recovery checks on seeded linear-Gaussian
DAGs: skeleton F1, v-structure recall, structural Hamming distance. PC
identifies only a Markov equivalence class — orientations beyond
v-structures + Meek closure are assumptions, not discovery. Never causal
evidence on real markets without a DAG identification argument.

Composition notes
-----------------
- ``metrics/dependence.py``: complementary dependence measures — this
  lane adds *conditional* independence tests and graph structure.
- ``models/causal_panel.py``: panel DID estimators — this lane learns
  the structure itself rather than estimating effects on a known graph.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _as_data(data: FloatArray, name: str = "data", min_rows: int = 30) -> FloatArray:
    a = np.asarray(data, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < min_rows or a.shape[1] < 2:
        raise ValueError(f"{name}: expected (n>={min_rows}, p>=2) matrix")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: non-finite values")
    return a


def partial_corr(i: int, j: int, cond: tuple[int, ...], data: FloatArray) -> tuple[float, float]:
    """Partial correlation of columns i,j given ``cond`` + Fisher-z p.

    Computes the correlation of residuals after OLS on ``cond`` columns;
    an empty ``cond`` returns Pearson correlation.
    """
    data = _as_data(data)
    n, p = data.shape
    if not (0 <= i < p and 0 <= j < p) or i == j:
        raise ValueError("i, j must be distinct column indices")
    if len(set(cond)) != len(cond) or any(c in (i, j) or c < 0 or c >= p for c in cond):
        raise ValueError("cond must be valid columns distinct from i, j")
    if len(cond) >= n - 4:
        raise ValueError("too few observations for this conditioning set")
    if cond:
        z = data[:, list(cond)]
        z = np.column_stack([np.ones(n), z])
        ri = data[:, i] - z @ np.linalg.lstsq(z, data[:, i], rcond=None)[0]
        rj = data[:, j] - z @ np.linalg.lstsq(z, data[:, j], rcond=None)[0]
    else:
        ri = data[:, i] - data[:, i].mean()
        rj = data[:, j] - data[:, j].mean()
    denom = float(np.sqrt((ri * ri).sum() * (rj * rj).sum()))
    if denom <= 0:
        return 0.0, 1.0
    r = float(np.clip((ri * rj).sum() / denom, -1.0, 1.0))
    dof = n - len(cond) - 3
    if dof <= 0:
        return r, 1.0
    if abs(r) >= 1.0 - 1e-12:
        return r, 0.0
    zstat = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
    pval = float(2.0 * stats.norm.sf(abs(zstat)))
    return r, pval


@dataclass(frozen=True)
class PCResult:
    skeleton: NDArray[np.bool_]
    cpdag: FloatArray  # +1 i->j, -1 i-j undirected, 0 absent
    sepsets: dict[tuple[int, int], tuple[int, ...]]
    n_tests: int


def pc_skeleton(
    data: FloatArray, alpha: float = 0.05, max_cond: int | None = None
) -> tuple[NDArray[np.bool_], dict[tuple[int, int], tuple[int, ...]], int]:
    """Stable PC skeleton: edge i-j removed if i ⊥ j | S for some S ⊆ adj.

    Order-independent variant (Colombo-Maathuis): adjacency sets are
    fixed at each level ℓ before any removals.
    """
    data = _as_data(data)
    n, p = data.shape
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if max_cond is None:
        max_cond = p - 2
    adj = np.ones((p, p), dtype=bool)
    np.fill_diagonal(adj, False)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    n_tests = 0
    ell = 0
    while ell <= max_cond:
        # snapshot adjacency for order-independence
        adj_level = adj.copy()
        changed = False
        for i, j in itertools.combinations(range(p), 2):
            if not adj[i, j]:
                continue
            nbrs_i = sorted(set(np.flatnonzero(adj_level[i]).tolist()) - {j})
            nbrs_j = sorted(set(np.flatnonzero(adj_level[j]).tolist()) - {i})
            removed = False
            for pool in (nbrs_i, nbrs_j):
                if len(pool) < ell:
                    continue
                for s in itertools.combinations(pool, ell):
                    if len(s) + 4 > n:
                        break
                    _r, pval = partial_corr(i, j, s, data)
                    n_tests += 1
                    if pval > alpha:
                        adj[i, j] = adj[j, i] = False
                        sepsets[(i, j)] = tuple(s)
                        removed = True
                        changed = True
                        break
                if removed:
                    break
        ell += 1
        if not changed and ell > max(int(np.max(adj.sum(axis=1))), 1):
            break
    return adj, sepsets, n_tests


def orient_vstructures(
    adj: NDArray[np.bool_], sepsets: dict[tuple[int, int], tuple[int, ...]]
) -> FloatArray:
    """Orient unshielded triples i-k-j as colliders when k ∉ sepset(i,j).

    Output ``cpdag[i,j] = +1`` means i→j; ``-1`` means undirected i—j.
    """
    p = adj.shape[0]
    cp = -np.asarray(adj, dtype=np.float64)  # undirected = -1
    np.fill_diagonal(cp, 0.0)
    for k in range(p):
        nbrs = np.flatnonzero(adj[k])
        for i, j in itertools.combinations(nbrs, 2):
            if adj[i, j]:  # shielded
                continue
            lo, hi = (int(i), int(j)) if i < j else (int(j), int(i))
            sepset = sepsets.get((lo, hi), sepsets.get((int(i), int(j)), ()))
            if k not in sepset:
                cp[i, k] = cp[k, i] = 0.0
                cp[i, k] = 1.0  # i -> k
                cp[k, i] = 0.0
                cp[j, k] = 1.0  # j -> k
                cp[k, j] = 0.0
    return cp


def apply_meek_rules(cp: FloatArray, max_rounds: int = 50) -> FloatArray:
    """Meek rules 1-3 until closure on a CPDAG matrix (+1 dir, -1 undir)."""
    cp = cp.copy()
    p = cp.shape[0]

    def undirected(i: int, j: int) -> bool:
        return bool(cp[i, j] == -1)

    def directed(i: int, j: int) -> bool:
        return bool(cp[i, j] == 1)

    def adjacent(i: int, j: int) -> bool:
        return bool(cp[i, j] != 0)

    for _ in range(max_rounds):
        changed = False
        for a, b in itertools.permutations(range(p), 2):
            if not undirected(a, b):
                continue
            # R1: exists c with c->a and c not adj b → a->b
            for c in range(p):
                if c in (a, b) or not directed(c, a) or adjacent(c, b):
                    continue
                cp[a, b] = 1.0
                changed = True
                break
            if changed:
                continue
            # R2: exists c with a->c->b → a->b
            for c in range(p):
                if c in (a, b) or not (directed(a, c) and directed(c, b)):
                    continue
                cp[a, b] = 1.0
                changed = True
                break
            if changed:
                continue
            # R3: a-c, a-d, c->b, d->b, c,d nonadjacent → a->b
            for c, d in itertools.combinations(range(p), 2):
                if c in (a, b) or d in (a, b):
                    continue
                if (
                    undirected(a, c)
                    and undirected(a, d)
                    and directed(c, b)
                    and directed(d, b)
                    and not adjacent(c, d)
                ):
                    cp[a, b] = 1.0
                    changed = True
                    break
        if not changed:
            break
    return cp


def pc_fit(data: FloatArray, alpha: float = 0.05) -> PCResult:
    """Skeleton + v-structure orientation + Meek closure."""
    adj, sepsets, n_tests = pc_skeleton(data, alpha)
    cp = orient_vstructures(adj, sepsets)
    cp = apply_meek_rules(cp)
    return PCResult(skeleton=adj, cpdag=cp, sepsets=sepsets, n_tests=n_tests)


def _bic_local(data: FloatArray, j: int, parents: tuple[int, ...]) -> float:
    """Gaussian BIC for column j regressed on parents: n log rss/n + k log n."""
    n = data.shape[0]
    y = data[:, j]
    if parents:
        xp = np.column_stack([np.ones(n)] + [data[:, c] for c in parents])
        beta = np.linalg.lstsq(xp, y, rcond=None)[0]
        resid = y - xp @ beta
    else:
        resid = y - y.mean()
    rss = float((resid * resid).sum()) + 1e-12
    k = len(parents) + 1
    return float(n * math.log(rss / n) + k * math.log(n))


def ges_fit(data: FloatArray, max_iter: int = 500) -> NDArray[np.bool_]:
    """Greedy forward-edge search by Gaussian BIC gain (single-edge adds).

    Simplified GES: at each step evaluate every candidate directed edge
    i→j (i not already a parent of j) that keeps the graph acyclic and
    strictly improves total BIC; apply the best gain. Documents itself as
    a forward-phase approximation of Chickering's two-phase search.
    """
    data = _as_data(data)
    n, p = data.shape
    parents: list[set[int]] = [set() for _ in range(p)]

    def creates_cycle(i: int, j: int) -> bool:
        # adding i->j creates a cycle iff j already reaches i via children
        seen = {j}
        stack = [j]
        while stack:
            u = stack.pop()
            for v in range(p):
                if u in parents[v] and v not in seen:  # u -> v edge
                    if v == i:
                        return True
                    seen.add(v)
                    stack.append(v)
        return False

    def total_bic() -> float:
        return sum(_bic_local(data, j, tuple(sorted(parents[j]))) for j in range(p))

    for _ in range(max_iter):
        best = (0.0, -1, -1)
        for i, j in itertools.permutations(range(p), 2):
            if i in parents[j] or creates_cycle(i, j):
                continue
            cur = _bic_local(data, j, tuple(sorted(parents[j])))
            new = _bic_local(data, j, tuple(sorted(parents[j] | {i})))
            gain = cur - new
            if gain > best[0] + 1e-9:
                best = (gain, i, j)
        if best[1] < 0:
            break
        parents[best[2]].add(best[1])
    adj = np.zeros((p, p), dtype=bool)
    for j, par in enumerate(parents):
        for i in par:
            adj[i, j] = True
    return adj


def lingam_pairwise(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Pairwise LiNGAM-lite direction score between two series.

    Uses the third-order cumulant asymmetry of the LiNGAM model: for
    x→y with non-Gaussian noise, ``E[x²y] - 2ρ E[x]E[xy]...`` — the
    Hyvärinen-Smith cumulant measure ``C = E[xy²]E[x²]-...`` simplified
    here to the sign of ``mean(x³y) - mean(xy³)``-style skew contrast
    ``R = corr(x,y) vs corr(x_resid, y_resid)`` after whitening. Returns
    direction (+1 x→y, -1 y→x) and score = asymmetry magnitude.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 30:
        raise ValueError("x, y: equal length >= 30")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("non-finite values")
    xs = (x - x.mean()) / max(x.std(), 1e-12)
    ys = (y - y.mean()) / max(y.std(), 1e-12)
    # Hyvärinen-Smith pairwise measure: E[x y^3] - E[x^3 y] flips sign
    # with direction under non-Gaussian noise.
    score = float(np.mean(xs * ys**3) - np.mean(xs**3 * ys))
    direction = 1.0 if score > 0 else -1.0
    return {"direction": direction, "score": abs(score)}


def synth_dag(
    n: int = 400,
    p: int = 6,
    edge_prob: float = 0.4,
    weight_range: tuple[float, float] = (0.5, 2.0),
    seed: int = 0,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Linear-Gaussian data on a random DAG (topological order = column order)."""
    if n < 30 or p < 2:
        raise ValueError("need n>=30, p>=2")
    if not (0 < edge_prob <= 1):
        raise ValueError("edge_prob in (0,1]")
    rng = np.random.default_rng(seed)
    adj = np.zeros((p, p), dtype=bool)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < edge_prob:
                adj[i, j] = True
    w = rng.uniform(*weight_range, size=(p, p)) * rng.choice([-1.0, 1.0], size=(p, p))
    w = np.where(adj, w, 0.0)
    x = np.empty((n, p))
    for j in range(p):
        noise = rng.standard_normal(n)
        x[:, j] = noise + x[:, :j] @ w[:j, j]
    return x, adj


def _skeleton_f1(pred: NDArray[np.bool_], true: NDArray[np.bool_]) -> float:
    tp = float(np.logical_and(pred, true).sum() / 2)
    fp = float(np.logical_and(pred, ~true).sum() / 2)
    fn = float(np.logical_and(~pred, true).sum() / 2)
    if tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn)


def _vstruct_recall(cp: FloatArray, true_adj: NDArray[np.bool_]) -> float:
    """Recall of true colliders i→k←j (i,j nonadjacent) in cpdag."""
    p = true_adj.shape[0]
    found = 0
    total = 0
    for k in range(p):
        pars = np.flatnonzero(true_adj[:, k])
        for i, j in itertools.combinations(pars, 2):
            if true_adj[i, j] or true_adj[j, i]:
                continue
            total += 1
            if cp[i, k] == 1.0 and cp[j, k] == 1.0:
                found += 1
    return found / total if total else 1.0


def _shd(cp: FloatArray, true_adj: NDArray[np.bool_]) -> int:
    """Structural Hamming distance on skeleton + wrong-orientation."""
    p = true_adj.shape[0]
    pred_adj = cp != 0
    shd = int(np.logical_xor(pred_adj, true_adj).sum() // 2)
    for i in range(p):
        for j in range(i + 1, p):
            if true_adj[i, j] and cp[i, j] == 1.0:
                continue
            if true_adj[j, i] and cp[j, i] == 1.0:
                continue
            if (true_adj[i, j] or true_adj[j, i]) and cp[i, j] != 0:
                shd += 1  # present but mis-oriented counts half-consistent
    return shd


def bench_causal_discovery(seed: int = 20261231 + 152) -> dict[str, float]:
    """SYNTHETIC structure-recovery telemetry."""
    rng = np.random.default_rng(seed)
    sk_f1s, vrecs, shds, ges_gains = [], [], [], []
    for rep in range(6):
        x, true_adj = synth_dag(n=500, p=6, edge_prob=0.35, seed=seed + rep)  # noqa: B007
        res = pc_fit(x, alpha=0.05)
        sk_f1s.append(_skeleton_f1(res.skeleton, true_adj))
        vrecs.append(_vstruct_recall(res.cpdag, true_adj))
        shds.append(float(_shd(res.cpdag, true_adj)))
        ges_adj = ges_fit(x)
        bic_full = sum(_bic_local(x, j, tuple(np.flatnonzero(ges_adj[:, j]))) for j in range(6))
        bic_empty = sum(_bic_local(x, j, ()) for j in range(6))
        ges_gains.append(bic_empty - bic_full)
    # LiNGAM direction on non-Gaussian chain
    correct = 0
    trials = 30
    for _rep in range(trials):
        e1 = rng.uniform(-1, 1, 300)
        e2 = rng.uniform(-1, 1, 300)
        xs = e1
        ys = 0.8 * xs + e2
        r = lingam_pairwise(xs, ys)
        if r["direction"] == 1.0:
            correct += 1
    res_rep = pc_fit(synth_dag(n=500, p=5, edge_prob=0.3, seed=seed + 999)[0], alpha=0.05)
    res_rep2 = pc_fit(synth_dag(n=500, p=5, edge_prob=0.3, seed=seed + 999)[0], alpha=0.05)
    deterministic = float(np.array_equal(res_rep.cpdag, res_rep2.cpdag))
    return {
        "synthetic_skeleton_f1": float(np.mean(sk_f1s)),
        "synthetic_vstructure_recall": float(np.mean(vrecs)),
        "synthetic_shd_mean": float(np.mean(shds)),
        "synthetic_ges_bic_gain": float(np.mean(ges_gains)),
        "synthetic_lingam_dir_acc": correct / trials,
        "synthetic_n_tests_mean": float(res_rep.n_tests),
        "synthetic_determinism": deterministic,
    }
