"""Held-Karp 1-tree lower bound for the TSP via subgradient ascent (SYNTHETIC).

A 1-tree = MST on nodes 2..n plus the two cheapest edges at node 1;
penalties pi_i adjust edge costs so the bound approaches the tour.
Subgradient ascent on degree violations; bench reports the best bound
vs the brute-force optimal tour (8 cities) and a 2-opt upper bound.
"""

import itertools

import numpy as np

from quant_fund.models._ilp_synth import HK_D


def _mst(d: np.ndarray, nodes: list[int]) -> tuple[float, list[tuple[int, int]]]:
    in_tree = {nodes[0]}
    edges = []
    total = 0.0
    while len(in_tree) < len(nodes):
        best = (np.inf, -1, -1)
        for i in in_tree:
            for j in nodes:
                if j not in in_tree and d[i, j] < best[0]:
                    best = (d[i, j], i, j)
        total += best[0]
        edges.append((best[1], best[2]))
        in_tree.add(best[2])
    return total, edges


def _one_tree(d: np.ndarray) -> tuple[float, dict[int, int]]:
    n = d.shape[0]
    rest = list(range(1, n))
    mst_cost, edges = _mst(d, rest)
    # two cheapest edges incident to node 0
    e0 = sorted((d[0, j], j) for j in rest)[:2]
    total = mst_cost + sum(w for w, _ in e0)
    deg = {i: 0 for i in range(n)}
    deg[0] = 2
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    for _, j in e0:
        deg[j] += 1
    return total, deg


def _held_karp(iters: int = 400) -> tuple[float, int]:
    n = HK_D.shape[0]
    pi = np.zeros(n)
    best, best_it = -np.inf, 0
    for it in range(iters):
        d = HK_D + pi[None, :] + pi[:, None]
        np.fill_diagonal(d, 0.0)
        w, deg = _one_tree(d)
        bound = w - 2.0 * pi.sum()
        if bound > best:
            best, best_it = bound, it
        g = np.array([deg[i] - 2 for i in range(n)], dtype=float)
        step = 1.0 / (it // 30 + 2)
        pi = pi + step * g
    return best, best_it


def _tour_brute() -> float:
    n = HK_D.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(1, n)):
        c = HK_D[0, perm[0]] + HK_D[0, perm[-1]]
        c += sum(HK_D[perm[i], perm[i + 1]] for i in range(n - 2))
        best = min(best, c)
    return float(best)


def _two_opt() -> float:
    n = HK_D.shape[0]
    tour = [0] + list(np.random.default_rng(3).permutation(range(1, n)).tolist())

    def cost(t: list[int]) -> float:
        return float(sum(HK_D[t[i], t[(i + 1) % n]] for i in range(n)))

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cand = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                if cost(cand) < cost(tour) - 1e-9:
                    tour, improved = cand, True
    return cost(tour)


def bench_held_karp(seed: int = 5111) -> dict[str, float]:
    bound, it_best = _held_karp()
    truth = _tour_brute()
    ub = _two_opt()
    return {
        "synthetic_hk_bound": bound,
        "synthetic_hk_truth": truth,
        "synthetic_hk_gap": float(truth - bound),
        "synthetic_hk_gap_frac": float((truth - bound) / truth),
        "synthetic_hk_ub": ub,
        "synthetic_hk_ub_gap": float(ub - truth),
        "synthetic_hk_best_iter": float(it_best),
    }
