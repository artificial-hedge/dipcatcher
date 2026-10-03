"""Shared fixtures for the wave-212 approximation-algorithms canon.

Small deterministic instances so every approximate optimum can be
checked against brute force:

- ``G_EDGES`` / ``G_N`` — undirected graph for vertex cover + max-cut
- ``KS_V`` / ``KS_W`` / ``KS_CAP`` — knapsack for the FPTAS
- set cover + TSP fixtures come from ``_ilp_synth``.
"""

import numpy as np

G_N = 10
G_EDGES = [
    (0, 1),
    (0, 2),
    (0, 5),
    (1, 2),
    (1, 3),
    (1, 8),
    (2, 4),
    (2, 9),
    (3, 4),
    (3, 8),
    (4, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (6, 7),
    (6, 9),
    (7, 8),
    (7, 9),
    (8, 9),
    (3, 9),
]

KS_V = np.array([13.0, 21.0, 17.0, 8.0, 31.0, 11.0, 26.0, 19.0])
KS_W = np.array([4.0, 7.0, 6.0, 3.0, 9.0, 5.0, 8.0, 6.0])
KS_CAP = 22.0


def brute_vc(edges: list[tuple[int, int]], n: int) -> int:
    import itertools

    for k in range(n + 1):
        for comb in itertools.combinations(range(n), k):
            cover = set(comb)
            if all(u in cover or v in cover for u, v in edges):
                return k
    return n


def brute_maxcut(edges: list[tuple[int, int]], n: int) -> int:
    best = 0
    for mask in range(1 << n):
        c = sum((mask >> u & 1) != (mask >> v & 1) for u, v in edges)
        best = max(best, c)
    return best


def brute_knapsack(v: np.ndarray, w: np.ndarray, cap: float) -> float:
    import itertools

    best = 0.0
    for xs in itertools.product([0, 1], repeat=len(v)):
        x = np.array(xs)
        if w @ x <= cap + 1e-9:
            best = max(best, float(v @ x))
    return best
