"""Connectedness and components on finite topologies (SYNTHETIC)."""

from __future__ import annotations

Topo = frozenset[frozenset[int]]


def clopens(opens: Topo, univ: frozenset[int]) -> frozenset[frozenset[int]]:
    return frozenset(o for o in opens if univ - o in opens)


def is_connected(univ: frozenset[int], opens: Topo) -> bool:
    """Connected iff only clopen sets are empty and whole."""
    c = clopens(opens, univ)
    return c <= frozenset({frozenset(), univ})


def components(univ: frozenset[int], opens: Topo) -> frozenset[frozenset[int]]:
    """Connected components via union-find on non-separable pairs."""
    parent = {x: x for x in univ}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # points x,y are together iff no clopen contains exactly one of them
    c = clopens(opens, univ)
    xs = list(univ)
    for i, x in enumerate(xs):
        for y in xs[i + 1 :]:
            if all((x in s) == (y in s) for s in c):
                parent[find(x)] = find(y)
    comps: dict[int, set[int]] = {}
    for x in univ:
        comps.setdefault(find(x), set()).add(x)
    return frozenset(frozenset(v) for v in comps.values())


def _bench_connected_space(seed: int = 0) -> float:
    checks = []
    u = frozenset({0, 1})
    discrete = frozenset({frozenset(), frozenset({0}), frozenset({1}), u})
    checks.append(not is_connected(u, discrete))
    indiscrete = frozenset({frozenset(), u})
    checks.append(is_connected(u, indiscrete))
    sier = frozenset({frozenset(), frozenset({0}), u})
    checks.append(is_connected(u, sier))
    comps = components(u, discrete)
    checks.append(len(comps) == 2)
    comps2 = components(u, sier)
    checks.append(len(comps2) == 1)
    checks.append(clopens(discrete, u) == discrete)
    return float(sum(checks) / len(checks))


def bench_connected_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_connected_space": _bench_connected_space(seed)}
