"""Graph coloring: greedy bound + exact chromatic number on small graphs (SYNTHETIC)."""

from __future__ import annotations

import itertools

Graph = dict[int, frozenset[int]]


def greedy_color(g: Graph, order: list[int] | None = None) -> dict[int, int]:
    vs = order if order is not None else sorted(g)
    color: dict[int, int] = {}
    for v in vs:
        used = {color[w] for w in g.get(v, frozenset()) if w in color}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def is_proper(g: Graph, color: dict[int, int]) -> bool:
    return all(color[v] != color[w] for v in g for w in g[v] if v in color and w in color)


def chromatic_number(g: Graph) -> int:
    vs = sorted(g)
    for k in range(1, len(vs) + 1):
        for assign in itertools.product(range(k), repeat=len(vs)):
            c = dict(zip(vs, assign, strict=True))
            if is_proper(g, c):
                return k
    return len(vs)


def max_degree(g: Graph) -> int:
    return max((len(n) for n in g.values()), default=0)


def _bench_graph_coloring(seed: int = 0) -> float:
    checks = []
    k3 = {0: frozenset({1, 2}), 1: frozenset({0, 2}), 2: frozenset({0, 1})}
    checks.append(chromatic_number(k3) == 3)
    path = {0: frozenset({1}), 1: frozenset({0, 2}), 2: frozenset({1})}
    checks.append(chromatic_number(path) == 2)
    c5 = {i: frozenset({(i - 1) % 5, (i + 1) % 5}) for i in range(5)}
    checks.append(chromatic_number(c5) == 3)
    # greedy respects Delta+1 bound on K4 (Delta=3, chi=4)
    k4 = {i: frozenset(j for j in range(4) if j != i) for i in range(4)}
    gc = greedy_color(k4)
    checks.append(len(set(gc.values())) <= max_degree(k4) + 1)
    checks.append(is_proper(k4, gc))
    checks.append(chromatic_number(k4) == 4)
    return float(sum(checks) / len(checks))


def bench_graph_coloring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_graph_coloring": _bench_graph_coloring(seed)}
