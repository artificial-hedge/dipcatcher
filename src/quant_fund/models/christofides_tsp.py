"""Christofides 1.5-approximation for metric TSP.

MST + min-weight perfect matching on odd-degree vertices + Eulerian
circuit shortcut to a tour. Matching on the (<=8) odd vertices by
brute-force pairings. Bench reports the tour vs the Held-Karp-style
brute-force optimum on the shared city-distance fixture.
"""

import numpy as np

from quant_fund.models._ilp_synth import HK_D
from quant_fund.models.held_karp import _mst, _tour_brute


def _perfect_matching(odd: list[int]) -> float:
    if not odd:
        return 0.0

    def rec(vs: tuple[int, ...]) -> float:
        if len(vs) <= 1:
            return 0.0
        v = vs[0]
        return float(min(HK_D[v, w] + rec(tuple(x for x in vs[1:] if x != w)) for w in vs[1:]))

    return rec(tuple(sorted(odd)))


def _christofides() -> tuple[float, int]:
    n = HK_D.shape[0]
    mst_cost, edges = _mst(HK_D, list(range(n)))
    deg = np.zeros(n, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    odd = [i for i in range(n) if deg[i] % 2 == 1]
    _perfect_matching(odd)  # bound check: matching is solved twice, small n
    # Euler multigraph -> walk -> shortcut: emulate by Euler circuit on
    # MST+matching adjacency
    adj: dict[int, list[int]] = {i: list() for i in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    # matching edges

    def match_pairs(vs):
        if len(vs) <= 1:
            yield []
        else:
            v = vs[0]
            for w in vs[1:]:
                for rest in match_pairs([x for x in vs[1:] if x != w]):
                    yield [(v, w)] + rest

    best_pairing = min(match_pairs(sorted(odd)), key=lambda ps: sum(HK_D[a, b] for a, b in ps))
    for a, b in best_pairing:
        adj[a].append(b)
        adj[b].append(a)
    # Hierholzer
    stack = [0]
    circuit: list[int] = []
    local = {k: list(v) for k, v in adj.items()}
    while stack:
        v = stack[-1]
        if local[v]:
            stack.append(local[v].pop())
            u = stack[-1]
            local[u].remove(v)
        else:
            circuit.append(stack.pop())
    tour = list(dict.fromkeys(circuit))
    cost = sum(HK_D[tour[i], tour[(i + 1) % n]] for i in range(len(tour)))
    return float(cost), len(odd)


def bench_christofides_tsp(seed: int = 5211) -> dict[str, float]:
    tour_cost, n_odd = _christofides()
    truth = _tour_brute()
    return {
        "synthetic_chr_tour": tour_cost,
        "synthetic_chr_truth": truth,
        "synthetic_chr_ratio": tour_cost / truth,
        "synthetic_chr_odd": float(n_odd),
    }
