"""2-SAT via implication-graph strongly connected components
(Kosaraju). Bench on random 2-CNF: solve rate vs brute force, and
recovery of a satisfying assignment when one exists.
"""

import itertools

import numpy as np


def _rand_2sat(rng: np.random.Generator, n_var: int, m: int) -> list[list[int]]:
    return [
        [
            int(v) * int(s)
            for v, s in zip(
                rng.choice(n_var, 2, replace=False) + 1, rng.choice([-1, 1], 2), strict=True
            )
        ]
        for _ in range(m)
    ]


def _scc_solve(clauses: list[list[int]], n_var: int) -> tuple[bool, dict[int, bool]]:
    # node ids: var v -> 2v (false lit), 2v+1 (true lit); lit lit>0 -> 2l+1, lit<0 -> 2|lit|
    def node(lit: int) -> int:
        return 2 * abs(lit) + (1 if lit > 0 else 0)

    def neg(lit: int) -> int:
        return 2 * abs(lit) + (0 if lit > 0 else 1)

    g: list[list[int]] = [[] for _ in range(2 * (n_var + 1) + 2)]
    for a, b in clauses:
        g[neg(a)].append(node(b))
        g[neg(b)].append(node(a))
    # Kosaraju
    order: list[int] = []
    seen = [False] * len(g)

    def dfs1(v: int) -> None:
        seen[v] = True
        for w in g[v]:
            if not seen[w]:
                dfs1(w)
        order.append(v)

    for v in range(len(g)):
        if not seen[v]:
            dfs1(v)
    comp = [-1] * len(g)
    gt: list[list[int]] = [[] for _ in g]
    for v in range(len(g)):
        for w in g[v]:
            gt[w].append(v)
    label = 0
    for v in reversed(order):
        if comp[v] != -1:
            continue
        stack = [v]
        comp[v] = label
        while stack:
            x = stack.pop()
            for w in gt[x]:
                if comp[w] == -1:
                    comp[w] = label
                    stack.append(w)
        label += 1
    for v in range(1, n_var + 1):
        if comp[node(v)] == comp[neg(v)]:
            return False, {}
    assign = {v: comp[node(v)] > comp[neg(v)] for v in range(1, n_var + 1)}
    return True, assign


def bench_twosat_scc(seed: int = 5907) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_var, m = 8, 10
    trials = 40
    agree = 0
    valid = 0
    for _ in range(trials):
        cl = _rand_2sat(rng, n_var, m)
        sat, a = _scc_solve(cl, n_var)
        truth = any(
            all(any(vals[abs(lit) - 1] == (lit > 0) for lit in c) for c in cl)
            for vals in itertools.product([False, True], repeat=n_var)
        )
        agree += int(sat == truth)
        if sat:
            valid += int(all(any(a[abs(lit)] == (lit > 0) for lit in c) for c in cl))
    return {
        "synthetic_2sat_agree": agree / trials,
        "synthetic_2sat_valid": valid / max(agree, 1),
    }
