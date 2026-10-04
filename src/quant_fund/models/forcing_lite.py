"""Forcing on finite posets: dense sets, generic filters (SYNTHETIC)."""

from __future__ import annotations

import itertools


def is_filter(poset_order: dict[tuple[int, int], bool], f: frozenset[int]) -> bool:
    """Filter: upward closed + any two members have common member of f below-join...
    Standard: for p,q in F exists r in F with r <= p, r <= q; and p in F, p <= q -> q in F."""
    for p in f:
        for q in f:
            # compatible inside F
            if not any(
                r in f and poset_order.get((r, p), False) and poset_order.get((r, q), False)
                for r in f
            ):
                return False
    # upward closure
    for p in f:
        for x in {b for (_a, b) in poset_order if poset_order.get((p, b), False)}:
            if x not in f:
                return False
    return True


def is_dense(
    poset_elems: frozenset[int], poset_order: dict[tuple[int, int], bool], d: frozenset[int]
) -> bool:
    """Dense: every element has a <= element inside D."""
    return all(any(poset_order.get((r, p), False) for r in d) for p in poset_elems)


def generic_filter(
    poset_elems: frozenset[int],
    poset_order: dict[tuple[int, int], bool],
    dense_sets: list[frozenset[int]],
) -> frozenset[int] | None:
    """Find a filter meeting every dense set (finite search over subsets)."""
    elems = sorted(poset_elems)
    for r in range(1, len(elems) + 1):
        for comb in itertools.combinations(elems, r):
            f = frozenset(comb)
            if is_filter(poset_order, f) and all(f & d for d in dense_sets):
                return f
    return None


def _bench_forcing_lite(seed: int = 0) -> float:
    checks = []
    # poset: 2-element antichain + top? Use order <= on {0,1}: 0 <= 1
    order = {(0, 0): True, (0, 1): True, (1, 1): True}
    checks.append(is_filter(order, frozenset({1})))
    checks.append(not is_filter(order, frozenset({0})))  # not upward closed
    checks.append(is_dense(frozenset({0, 1}), order, frozenset({0})))
    checks.append(
        not is_dense(frozenset({0, 1}), order, frozenset({1}))
    )  # 0 has no <= elt in {1}? 0<=0 only -> dense? D={1}: need r in D, r<=0 -> none -> not dense
    g = generic_filter(frozenset({0, 1}), order, [frozenset({0})])
    checks.append(bool(g is not None and 0 in g))
    return float(sum(checks) / len(checks))


def bench_forcing_lite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forcing_lite": _bench_forcing_lite(seed)}
