"""Order dimension: smallest realizer of linear extensions (SYNTHETIC)."""

from __future__ import annotations

import itertools

Poset = frozenset[tuple[int, int]]  # covering relations a < b


def is_linear_extension(rels: Poset, elems: frozenset[int], order: list[int]) -> bool:
    """All relations respected by the linear order."""
    pos = {x: i for i, x in enumerate(order)}
    return all(pos[a] < pos[b] for a, b in rels)


def realizer_size(rels: Poset, elems: frozenset[int], k: int) -> bool:
    """Exists k linear orders whose intersection = the poset order."""
    vs = sorted(elems)
    exts = [
        list(p) for p in itertools.permutations(vs) if is_linear_extension(rels, elems, list(p))
    ]
    incomparable = [
        (a, b) for a in vs for b in vs if a != b and (a, b) not in rels and (b, a) not in rels
    ]
    for combo in itertools.combinations(exts, k):
        ok = True
        for a, b in incomparable:
            if not any(o.index(a) > o.index(b) for o in combo):
                ok = False
                break
        if ok:
            return True
    return False


def order_dimension(rels: Poset, elems: frozenset[int]) -> int:
    for k in range(1, len(elems) + 1):
        if realizer_size(rels, elems, k):
            return k
    return len(elems)


def _bench_poset_dimension(seed: int = 0) -> float:
    checks = []
    # total order: dim 1
    chain = frozenset({(0, 1), (1, 2), (0, 2)})
    checks.append(order_dimension(chain, frozenset({0, 1, 2})) == 1)
    # antichain: dim 2 (two orders suffice for any antichain >1)
    checks.append(order_dimension(frozenset(), frozenset({0, 1})) == 2)
    # N poset: a<b, c<b — dim 2
    npos = frozenset({(0, 2), (1, 2)})
    checks.append(order_dimension(npos, frozenset({0, 1, 2})) == 2)
    # crown/standard example S3: dim 3? use dim check on 4-cycle poset {a<b,a<d,c<b,c<d}: dim 2
    crown = frozenset({(0, 2), (0, 3), (1, 2), (1, 3)})
    checks.append(order_dimension(crown, frozenset({0, 1, 2, 3})) == 2)
    checks.append(is_linear_extension(chain, frozenset({0, 1, 2}), [0, 1, 2]))
    checks.append(not is_linear_extension(chain, frozenset({0, 1, 2}), [1, 0, 2]))
    return float(sum(checks) / len(checks))


def bench_poset_dimension(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poset_dimension": _bench_poset_dimension(seed)}
