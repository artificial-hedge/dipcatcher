"""Fundamental group of graphs: pi1 = free group on non-spanning-tree edges (SYNTHETIC)."""

from __future__ import annotations


def spanning_tree(n: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """BFS tree edges."""
    adj: dict[int, list[tuple[int, tuple[int, int]]]] = {v: [] for v in range(n)}
    for a, b in edges:
        adj[a].append((b, (a, b)))
        adj[b].append((a, (a, b)))
    seen = {0}
    tree = []
    stack = [0]
    while stack:
        u = stack.pop()
        for w, e in adj[u]:
            if w not in seen:
                seen.add(w)
                tree.append(e)
                stack.append(w)
    return tree


def pi1_rank(n: int, edges: list[tuple[int, int]]) -> int:
    """Free rank = E - (V-1) for connected graph."""
    tree = spanning_tree(n, edges)
    return len(edges) - len(tree)


def loop_reduce(word: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Free reduction: cancel adjacent inverse generators."""
    out: list[tuple[str, int]] = []
    for g, e in word:
        if out and out[-1][0] == g and out[-1][1] == -e:
            out.pop()
        else:
            out.append((g, e))
    return out


def _bench_fundamental_group(seed: int = 0) -> float:
    checks = []
    # cycle graph C3: pi1 = Z -> rank 1
    c3 = [(0, 1), (1, 2), (2, 0)]
    checks.append(pi1_rank(3, c3) == 1)
    # tree: rank 0
    checks.append(pi1_rank(3, [(0, 1), (1, 2)]) == 0)
    # figure-8: two loops sharing vertex -> rank 2
    checks.append(pi1_rank(3, [(0, 1), (1, 2), (2, 0), (0, 2)]) == 2)
    # free reduction
    checks.append(loop_reduce([("a", 1), ("a", -1), ("b", 1)]) == [("b", 1)])
    checks.append(loop_reduce([("a", 1), ("b", 1), ("b", -1), ("a", -1)]) == [])
    checks.append(loop_reduce([("a", 1), ("b", -1)]) == [("a", 1), ("b", -1)])
    return float(sum(checks) / len(checks))


def bench_fundamental_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fundamental_group": _bench_fundamental_group(seed)}
