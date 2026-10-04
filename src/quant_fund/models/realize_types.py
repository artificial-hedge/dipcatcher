"""Complete types over finite structures and their realization (SYNTHETIC)."""

from __future__ import annotations


def qf_type_of(
    elem: int,
    domain: list[int],
    lt,
    params: list[int] | None = None,
) -> frozenset[tuple[str, int]]:
    """Quantifier-free 1-type of `elem` over `params` in a finite order:
    the comparisons it makes with each parameter element."""
    if params is None:
        params = domain
    facts: set[tuple[str, int]] = set()
    for d in params:
        if d == elem:
            continue
        if lt(elem, d):
            facts.add(("lt", d))
        else:
            facts.add(("gt", d))
    return frozenset(facts)


def realize(
    conditions: frozenset[tuple[str, int]],
    domain: list[int],
    lt,
) -> int | None:
    """Find an element realizing the given partial type, if any."""
    for cand in domain:
        ok = True
        for op, d in conditions:
            if (
                op == "lt"
                and not lt(cand, d)
                or op == "gt"
                and not lt(d, cand)
                or op == "eq"
                and cand != d
            ):
                ok = False
            if not ok:
                break
        if ok:
            return cand
    return None


def _bench_realize_types(seed: int = 0) -> float:
    checks = []
    chain = [0, 1, 2, 3, 4]
    lt = lambda a, b: a < b  # noqa: E731
    # every position has a distinct type in a finite chain
    types = {qf_type_of(e, chain, lt) for e in chain}
    checks.append(len(types) == 5)
    # type of 2 says: lt with {3,4}, gt with {0,1}
    t2 = qf_type_of(2, chain, lt)
    checks.append(("lt", 3) in t2 and ("gt", 0) in t2 and len(t2) == 4)
    # realize "between 1 and 3" -> only 2
    conds = frozenset({("gt", 1), ("lt", 3)})
    checks.append(realize(conds, chain, lt) == 2)
    # unrealizable: gt 4 and lt 0
    checks.append(realize(frozenset({("gt", 4), ("lt", 0)}), chain, lt) is None)

    # over the empty parameter set (pure equality) every element of the
    # domain realizes the same 1-type {x = x}
    def lt2(a: int, b: int) -> bool:
        return False  # no order: all elements look alike

    ts = {qf_type_of(e, chain, lt2, params=[]) for e in chain}
    checks.append(len(ts) == 1)
    # realize partial type {gt 0, lt 2, gt ... } consistent -> 1
    checks.append(realize(frozenset({("gt", 0), ("lt", 2)}), chain, lt) == 1)
    return float(sum(checks) / len(checks))


def bench_realize_types(seed: int = 0) -> dict[str, float]:
    return {"synthetic_realize_types": _bench_realize_types(seed)}
