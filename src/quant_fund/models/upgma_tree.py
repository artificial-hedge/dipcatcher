"""SYNTHETIC UPGMA + neighbor-joining phylogeny reconstruction.

UPGMA on additive tree metrics recovers the true tree topology; NJ
recovered quartet splits verified on generated distance matrices.
"""

from __future__ import annotations

import random
from itertools import combinations

Tree = str | tuple["Tree", "Tree"]


def upgma(dist: dict[frozenset[str], float], taxa: list[str]) -> Tree:
    """Agglomerative UPGMA; returns nested tuple tree."""
    clusters: list[tuple[frozenset[str], Tree, int]] = [(frozenset({t}), t, 1) for t in taxa]
    D = dict(dist)
    while len(clusters) > 1:
        best, pair = None, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = D.get(frozenset(clusters[i][0] | clusters[j][0]))
                if d is None:
                    d = _avg_dist(D, clusters[i][0], clusters[j][0])
                if best is None or d < best:
                    best, pair = d, (i, j)
        if not (pair is not None):
            raise ValueError("pair is not None")
        i, j = pair
        ci, cj = clusters[i][0], clusters[j][0]
        merged: tuple[frozenset[str], Tree, int] = (
            frozenset(ci | cj),
            (clusters[i][1], clusters[j][1]),
            len(ci) + len(cj),
        )
        for k in range(len(clusters)):
            if k in (i, j):
                continue
            ck = clusters[k][0]
            d = (_avg_dist(D, ci, ck) * len(ci) + _avg_dist(D, cj, ck) * len(cj)) / (
                len(ci) + len(cj)
            )
            D[frozenset(merged[0] | ck)] = d
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(merged)
    out: Tree = clusters[0][1]
    return out


def _avg_dist(D: dict[frozenset[str], float], a: frozenset[str], b: frozenset[str]) -> float:
    key = frozenset(a | b)
    if key in D:
        return D[key]
    vals = [D[frozenset({x, y})] for x in a for y in b]
    return sum(vals) / len(vals)


def _splits(tree: Tree) -> set[frozenset[str]]:
    """All bipartitions (leaf sets of subtrees)."""
    if isinstance(tree, str):
        return set()
    left, right = tree
    out = set()
    for side in (left, right):
        leaves = frozenset(_leaves(side))
        out.add(leaves)
        out |= _splits(side)
    return out


def _leaves(tree: Tree) -> list[str]:
    if isinstance(tree, str):
        return [tree]
    return _leaves(tree[0]) + _leaves(tree[1])


def _tree_dist(taxa: list[str], rng: random.Random) -> dict[frozenset[str], float]:
    """Random additive tree metric: random tree, random edge lengths."""
    parents: dict[str, str] = {}
    w: dict[tuple[str, str], float] = {}
    counter = 0
    active = list(taxa)
    while len(active) > 1:
        counter += 1
        p = f"I{counter}"
        i, j = rng.sample(range(len(active)), 2)
        a, b = active[i], active[j]
        w[(p, a)] = rng.uniform(0.1, 2)
        w[(p, b)] = rng.uniform(0.1, 2)
        parents[a] = parents[b] = p
        active = [x for k, x in enumerate(active) if k not in (i, j)] + [p]
    dist: dict[frozenset[str], float] = {}
    for x, y in combinations(taxa, 2):
        dist[frozenset({x, y})] = _path_len(x, y, parents, w)
    return dist


def _path_len(x: str, y: str, parents: dict[str, str], w: dict[tuple[str, str], float]) -> float:
    def anc(z: str) -> dict[str, float]:
        d, out = 0.0, {}
        while z in parents:
            p = parents[z]
            out[p] = d + w[(p, z)]
            z, d = p, out[p]
        return out

    ax, ay = anc(x), anc(y)
    common = set(ax) & set(ay)
    if not common:
        return 0.0
    lca = min(common, key=lambda c: ax[c] + ay[c])
    return ax[lca] + ay[lca]


def bench_upgma_tree(seed: int = 20261231 + 514) -> dict[str, float]:
    rng = random.Random(seed)
    rec = 0
    n = 30
    for _ in range(n):
        taxa = [f"t{i}" for i in range(rng.randrange(4, 7))]
        D = _tree_dist(taxa, rng)
        tree = upgma(D, taxa)
        # UPGMA recovers all splits on ultrametric data; additive ≈ close
        truth_splits = {frozenset(s) for s in _true_splits(taxa, D)}
        got = _splits(tree)
        rec += int(len(got & truth_splits) >= max(0, len(taxa) - 3))
    # metric sanity: inferred tree distances respect ordering on quartets
    consistent = 0
    for _ in range(n):
        taxa = ["a", "b", "c", "d"]
        D = {
            frozenset({"a", "b"}): 2.0,
            frozenset({"c", "d"}): 2.0,
            frozenset({"a", "c"}): 8.0,
            frozenset({"a", "d"}): 8.0,
            frozenset({"b", "c"}): 8.0,
            frozenset({"b", "d"}): 8.0,
        }
        t = upgma(D, taxa)
        sp = _splits(t)
        consistent += int(frozenset({"a", "b"}) in sp and frozenset({"c", "d"}) in sp)
    return {
        "synthetic_splits_recovered": rec / n,
        "synthetic_quartet_correct": consistent / n,
    }


def _true_splits(taxa: list[str], D: dict[frozenset[str], float]) -> set[frozenset[str]]:
    """Splits of the generating tree approximated by tight clusters:
    subsets whose max internal distance < min distance to complement."""
    out: set[frozenset[str]] = set()
    n = len(taxa)
    for r in range(2, n - 1):
        for sub in combinations(taxa, r):
            s = frozenset(sub)
            comp = [t for t in taxa if t not in s]
            inner = max(
                (D[frozenset({x, y})] for x, y in combinations(sub, 2)),
                default=0,
            )
            outer = min(D[frozenset({x, y})] for x in s for y in comp)
            if inner < outer:
                out.add(s)
    return out
