"""Product topology on finite spaces + projection continuity (SYNTHETIC)."""

from __future__ import annotations

import itertools

Topo = frozenset[frozenset[int]]
PTopo = frozenset[frozenset[tuple[int, int]]]


def product_opens(ox: Topo, oy: Topo, xset: frozenset[int], yset: frozenset[int]) -> PTopo:
    """All unions of basic opens U x V."""
    basics = [frozenset((x, y) for x in u for y in v) for u in ox for v in oy]
    out: set[frozenset[tuple[int, int]]] = set()
    for r in range(len(basics) + 1):
        for combo in itertools.combinations(basics, r):
            out.add(frozenset().union(*combo) if combo else frozenset())
    return frozenset(out)


def pi1_cont(prod: PTopo, ox: Topo, xset: frozenset[int], yset: frozenset[int]) -> bool:
    """Projection pi1 continuous: preimage of each X-open is product-open."""
    for u in ox:
        pre = frozenset((x, y) for x in u for y in yset)
        if pre not in prod:
            return False
    return True


def _bench_product_topology(seed: int = 0) -> float:
    checks = []
    u = frozenset({0, 1})
    disc = frozenset(frozenset(c) for r in range(3) for c in itertools.combinations(u, r))
    ind = frozenset({frozenset(), u})
    prod_dd = product_opens(disc, disc, u, u)
    checks.append(len(prod_dd) == 16)  # discrete x discrete = discrete on 4 pts
    prod_di = product_opens(disc, ind, u, u)
    checks.append(len(prod_di) == 4)  # {empty, {0}x{0,1}, {1}x{0,1}, all}
    checks.append(pi1_cont(prod_dd, disc, u, u))
    checks.append(pi1_cont(prod_di, disc, u, u))
    return float(sum(checks) / len(checks))


def bench_product_topology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_product_topology": _bench_product_topology(seed)}
