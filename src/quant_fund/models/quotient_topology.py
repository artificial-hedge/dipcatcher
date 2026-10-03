"""Quotient topology on finite spaces + universal property (SYNTHETIC)."""

from __future__ import annotations

Topo = frozenset[frozenset[int]]


def quotient_opens(opens: Topo, quotient_map) -> Topo:
    """Opens of the quotient: sets U in Y such that f^{-1}(U) is open in X."""
    # collect images
    images = frozenset(quotient_map.values())
    subs = []
    ys = list(images)
    import itertools

    for r in range(len(ys) + 1):
        for c in itertools.combinations(ys, r):
            cand = frozenset(c)
            pre = frozenset(x for x, fx in quotient_map.items() if fx in cand)
            if pre in opens:
                subs.append(cand)
    return frozenset(subs)


def respects_quotient(opens_x: Topo, opens_y: Topo, quotient_map) -> bool:
    """f continuous: preimage of every Y-open is X-open."""
    for o in opens_y:
        pre = frozenset(x for x, fx in quotient_map.items() if fx in o)
        if pre not in opens_x:
            return False
    return True


def _bench_quotient_topology(seed: int = 0) -> float:
    checks = []
    import itertools

    u = frozenset({0, 1, 2})
    disc = frozenset(frozenset(c) for r in range(4) for c in itertools.combinations(u, r))
    # collapse {1,2} to a point: f(0)=0, f(1)=f(2)=1
    q = {0: 0, 1: 1, 2: 1}
    qo = quotient_opens(disc, q)
    # quotient opens: subsets of {0,1} whose preimage is open — all subsets qualify
    checks.append(len(qo) == 4)
    checks.append(respects_quotient(disc, qo, q))
    # indiscrete source: only whole-set preimages open
    ind = frozenset({frozenset(), u})
    qo2 = quotient_opens(ind, q)
    checks.append(qo2 == frozenset({frozenset(), frozenset({0, 1})}))
    return float(sum(checks) / len(checks))


def bench_quotient_topology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quotient_topology": _bench_quotient_topology(seed)}
