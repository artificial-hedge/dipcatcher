"""Difference logic (QF_DL): x - y <= c systems + Bellman-Ford consistency (SYNTHETIC bench)."""

from __future__ import annotations

INF = float("inf")


def consistent(constraints: list[tuple[int, int, float]], n_vars: int) -> bool:
    """x_i - x_j <= c constraints; inconsistent iff a negative cycle exists."""
    n = n_vars + 1
    src = n_vars  # virtual zero source
    edges = [(j, i, c) for i, j, c in constraints] + [(src, i, 0.0) for i in range(n_vars)]
    dist = [0.0] * n
    for _ in range(n - 1):
        for u, v, c in edges:
            if dist[u] + c < dist[v]:
                dist[v] = dist[u] + c
    return not any(dist[u] + c < dist[v] - 1e-9 for u, v, c in edges)


def tightest(constraints: list[tuple[int, int, float]], n_vars: int) -> list[list[float]]:
    """Bounds[i][j] = tightest implied upper bound on x_i - x_j (shortest paths)."""
    n = n_vars
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0.0
    for i, j, c in constraints:
        if c < d[i][j]:
            d[i][j] = c
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def implies(
    constraints: list[tuple[int, int, float]], n_vars: int, q: tuple[int, int, float]
) -> bool:
    """Does the system imply x_i - x_j <= c? (tightest bound <= c)."""
    if not consistent(constraints, n_vars):
        return True
    d = tightest(constraints, n_vars)
    return bool(d[q[0]][q[1]] <= q[2] + 1e-9)


def explain(constraints: list[tuple[int, int, float]], n_vars: int) -> list[int]:
    """Indices of constraints on a negative cycle (a minimal-ish UNSAT core)."""
    n = n_vars + 1
    src = n_vars
    edges = [(j, i, c, k) for k, (i, j, c) in enumerate(constraints)] + [
        (src, i, 0.0, -1) for i in range(n_vars)
    ]
    dist = [0.0] * n
    par = [-1] * n
    last = -1
    for _ in range(n):
        last = -1
        for u, v, c, _k in edges:
            if dist[u] + c < dist[v] - 1e-12:
                dist[v] = dist[u] + c
                par[v] = u
                last = v
    if last < 0:
        return []
    v = last
    for _ in range(n):
        v = par[v]
    core: list[int] = []
    u = v
    while True:
        for a, b, _, k in edges:
            if a == par[u] and b == u and k >= 0:
                core.append(k)
                break
        u = par[u]
        if u == v:
            break
    return core


def _bench_diff_logic(seed: int = 0) -> float:
    del seed
    checks = []
    checks.append(consistent([(0, 1, 5.0), (1, 2, 3.0)], 3))
    checks.append(not consistent([(0, 1, -2.0), (1, 0, 1.0)], 2))
    checks.append(implies([(0, 1, 5.0), (1, 2, 3.0)], 3, (0, 2, 8.0)))
    checks.append(not implies([(0, 1, 5.0)], 2, (0, 1, 4.0)))
    core = explain([(0, 1, -2.0), (1, 0, 1.0), (2, 0, 3.0)], 3)
    checks.append(sorted(core) == [0, 1])
    d = tightest([(0, 1, 5.0), (1, 2, 3.0)], 3)
    checks.append(abs(d[0][2] - 8.0) < 1e-9)
    return sum(checks) / len(checks)


def bench_diff_logic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diff_logic": _bench_diff_logic(seed)}
