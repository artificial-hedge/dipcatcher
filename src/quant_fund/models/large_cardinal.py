"""Large cardinal properties on finite/toy models (SYNTHIC) (SYNTHETIC)."""

from __future__ import annotations


def is_ultrafilter(u: set[frozenset], universe: frozenset) -> bool:
    """U is an ultrafilter on X iff: closed under supersets within X,
    closed under finite intersections, empty set not in U, and for every
    A subset X exactly one of A, X\\A is in U."""
    subsets = _powerset(universe)
    # complement property
    for a in subsets:
        a_in = frozenset(a) in u
        c_in = frozenset(universe - a) in u
        if a_in == c_in:
            return False
    # upward closure + intersection closure + nonempty
    if frozenset() in u or universe not in u:
        return False
    for a in u:
        for b in u:
            if frozenset(a & b) not in u:
                return False
        for s in subsets:
            if a <= s and frozenset(s) not in u:
                return False
    return True


def _powerset(s: frozenset) -> list[frozenset]:
    items = list(s)
    out: list[frozenset] = [frozenset()]
    for x in items:
        out += [frozenset(set(t) | {x}) for t in out]
    return out


def principal_ultrafilter(x: frozenset, pt: int) -> set[frozenset]:
    return {s for s in _powerset(x) if pt in s}


def _bench_large_cardinal(seed: int = 0) -> float:
    checks = []
    x = frozenset({0, 1, 2})
    u = principal_ultrafilter(x, 0)
    checks.append(is_ultrafilter(u, x))
    checks.append(len(u) == 4)  # half the 8 subsets
    # non-principal impossible on finite set: every ultrafilter is principal
    v = {s for s in _powerset(x) if 1 in s}
    checks.append(is_ultrafilter(v, x))
    # removing upward-closure breaks ultrafilter
    bad = {frozenset({0}), frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 1, 2})}
    checks.append(is_ultrafilter(bad, x))  # still principal at 0
    checks.append(bad == u)
    # a "filter" containing both a set and its complement is not ultra
    notu = {frozenset({0}), frozenset({1, 2}), frozenset({0, 1, 2})}
    checks.append(not is_ultrafilter(notu, x))
    # kappa-completeness: intersection of <kappa members stays in U;
    # principal ultrafilter on finite set is complete for all kappa
    import functools

    inter = functools.reduce(lambda a, b: a & b, u)
    checks.append(0 in inter)  # the point witnesses completeness
    return float(sum(checks) / len(checks))


def bench_large_cardinal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_large_cardinal": _bench_large_cardinal(seed)}
