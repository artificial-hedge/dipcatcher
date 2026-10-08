"""Kan composition (hcomp) on a finite graph model of paths (SYNTHETIC).

Elements are vertices; a path x ~ y is an edge-walk in the graph.
hcomp fills an open box: given a base vertex and side paths leaving it,
the composite face is the walk concatenation — checked for consistency
(sides must share their endpoint) which is the cubical condition.
"""

from __future__ import annotations

_SEED = 20261231 + 1056

Walk = tuple  # tuple of vertices; consecutive must be adjacent


def _adjacent(edges: set[frozenset], u, v) -> bool:
    return u == v or frozenset({u, v}) in edges


def is_walk(edges: set[frozenset], w: Walk) -> bool:
    return all(_adjacent(edges, w[i], w[i + 1]) for i in range(len(w) - 1))


def concat(w1: Walk, w2: Walk) -> Walk:
    if w1[-1] != w2[0]:
        raise ValueError("walks don't meet")
    return w1 + w2[1:]


def hcomp(
    edges: set[frozenset],
    base,
    sides: list[Walk],
) -> Walk | None:
    """Fill a 2D box: each side_i is a walk from base_i to corner_i.
    Kan condition: consecutive corners must connect. Returns composite
    walk from base to final corner or None if unfilled."""
    if not sides:
        return (base,)
    cur: Walk = (base,)
    for s in sides:
        if not is_walk(edges, s):
            return None
        if cur[-1] != s[0]:
            return None
        cur = concat(cur, s)
    return cur


def connected(edges: set[frozenset], u, v) -> bool:
    seen = {u}
    wl = [u]
    while wl:
        x = wl.pop()
        for e in edges:
            if x in e:
                y = next(iter(e - {x})) if len(e) == 2 else x
                if y not in seen:
                    seen.add(y)
                    wl.append(y)
    return v in seen


def bench_kan_hcomp(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # square: 0-1-2-3-0
    e = {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3}), frozenset({0, 3})}
    w = hcomp(e, 0, [(0, 1), (1, 2)])
    checks.append(w == (0, 1, 2))
    # disconnected sides can't fill
    checks.append(hcomp(e, 0, [(0, 1), (2, 3)]) is None)
    # non-walk side rejected
    checks.append(hcomp(e, 0, [(0, 2)]) is None)
    # full square composite: 0->1->2->3
    checks.append(hcomp(e, 0, [(0, 1), (1, 2), (2, 3)]) == (0, 1, 2, 3))
    # connectivity respects edges
    checks.append(connected(e, 0, 3) and not connected({frozenset({0, 1})}, 0, 3))
    # is_walk validates
    checks.append(is_walk(e, (0, 1, 2)) and not is_walk(e, (0, 2)))
    return {"synthetic_kan_hcomp": float(sum(checks)) / len(checks)}
