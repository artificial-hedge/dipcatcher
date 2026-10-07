"""Fitch small-parsimony score on a fixed tree (wave 284) (SYNTHETIC).

Set-intersection/union postorder gives the minimum number of state changes;
verified against brute-force ancestral-state search on small trees.
"""

import itertools

import numpy as np

_SEED = 20261231 + 789


def fitch(leaves: dict[str, int], edges: list[tuple[str, str]]) -> int:
    children: dict[str, list[str]] = {}
    for p, c in edges:
        children.setdefault(p, []).append(c)

    def rec(node: str) -> tuple[set[int], int]:
        if node in leaves:
            return {leaves[node]}, 0
        sets, cost = [], 0
        for ch in children[node]:
            s, c = rec(ch)
            sets.append(s)
            cost += c
        inter = set.intersection(*sets)
        if inter:
            return inter, cost
        return set.union(*sets), cost + 1

    all_nodes = set(children) | set(leaves)
    child_set = {c for _, c in edges}
    roots = [n for n in all_nodes if n not in child_set]
    return rec(roots[0])[1]


def _brute(leaves: dict[str, int], edges: list[tuple[str, str]], states: int) -> int:
    internal = [n for n in {p for p, _ in edges} if n not in leaves]
    best = 10**9
    for assn in itertools.product(range(states), repeat=len(internal)):
        lab = dict(leaves)
        lab.update(dict(zip(internal, assn, strict=True)))
        cost = sum(lab[p] != lab[c] for p, c in edges)
        best = min(best, cost)
    return best


def bench_fitch_pars(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(5):
        # rooted binary tree over 4 leaves: ((a,b),(c,d)) or ((a,c),(b,d))
        leaves = {k: int(rng.randint(0, 2)) for k in ["a", "b", "c", "d"]}
        edges = [("u", "a"), ("u", "b"), ("v", "c"), ("v", "d"), ("r", "u"), ("r", "v")]
        ok += int(fitch(leaves, edges) == _brute(leaves, edges, 2))
    return {"synthetic_fitch": float(ok == 5)}
