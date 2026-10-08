"""Shared synthetic fixtures + tiny simplex for the wave-211 IP canon (SYNTHETIC)."""

import numpy as np

# small ILP for gomory/branch-and-cut: max 5x+8y, x+y<=6, 5x+9y<=45
ILP_C = np.array([5.0, 8.0])
ILP_A = np.array([[1.0, 1.0], [5.0, 9.0]])
ILP_B = np.array([6.0, 45.0])
ILP_UB = np.array([8, 8])

# cutting stock: stock length 100, demand widths/counts
CS_W = 100.0
CS_WIDTHS = np.array([45.0, 36.0, 31.0, 14.0])
CS_DEMAND = np.array([97, 610, 395, 211])

# facility location: 5 candidate sites, 6 customers
FL_F = np.array([40.0, 50.0, 35.0, 55.0, 30.0])
FL_C = np.array(
    [
        [8, 10, 14, 6, 9],
        [11, 7, 9, 12, 10],
        [6, 12, 8, 9, 14],
        [13, 8, 11, 7, 6],
        [9, 14, 7, 11, 8],
        [12, 9, 13, 8, 10],
    ],
    dtype=float,
)

# set cover for Lagrangian: 8 sets over 10 elements
SC_C = np.array([4.0, 6.0, 3.0, 8.0, 5.0, 2.0, 7.0, 4.0])
SC_A = np.array(
    [
        [1, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1],
    ],
    dtype=float,
)

# TSP instance for Held-Karp 1-tree bound (8 cities)
HK_D = np.array(
    [
        [0, 29, 20, 21, 16, 31, 100, 12],
        [29, 0, 15, 29, 28, 40, 72, 21],
        [20, 15, 0, 15, 14, 25, 81, 9],
        [21, 29, 15, 0, 4, 12, 92, 12],
        [16, 28, 14, 4, 0, 16, 94, 9],
        [31, 40, 25, 12, 16, 0, 95, 24],
        [100, 72, 81, 92, 94, 95, 0, 90],
        [12, 21, 9, 12, 9, 24, 90, 0],
    ],
    dtype=float,
)


def simplex(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """max c'x, a x <= b, x >= 0 — tableau simplex with Bland's rule."""
    m, n = a.shape
    t = np.hstack([a.astype(float), np.eye(m), b.reshape(-1, 1)])
    basis = list(range(n, n + m))
    cost = np.concatenate([c, np.zeros(m + 1)])
    while True:
        c_b = cost[basis]
        reduced = cost[: n + m] - c_b @ t[:, : n + m]
        j = -1
        for cand in range(n + m):
            if reduced[cand] > 1e-9:
                j = cand
                break
        if j < 0:
            break
        col = t[:, j]
        if (col <= 1e-12).all():
            raise ValueError("unbounded")
        cand_rows = [i for i in range(m) if col[i] > 1e-12]
        i = min(cand_rows, key=lambda r: (t[r, -1] / col[r], basis[r]))
        t[i] = t[i] / t[i, j]
        for r in range(m):
            if r != i:
                t[r] = t[r] - t[r, j] * t[i]
        basis[i] = j
    x = np.zeros(n + m)
    x[basis] = t[:, -1]
    return x[:n], float(c @ x[:n])


def brute_set_cover(costs: np.ndarray, a: np.ndarray) -> float:
    import itertools

    best = np.inf
    n = a.shape[1]
    for xs in itertools.product([0, 1], repeat=n):
        x = np.array(xs, dtype=float)
        if (a @ x >= 1.0 - 1e-9).all():
            best = min(best, float(costs @ x))
    return best


def brute_force_ilp(c: np.ndarray, a: np.ndarray, b: np.ndarray, ub: np.ndarray) -> float:
    best = -np.inf
    import itertools

    for xs in itertools.product(*[range(u + 1) for u in ub]):
        x = np.array(xs, dtype=float)
        if (a @ x <= b + 1e-9).all():
            best = max(best, float(c @ x))
    return best
