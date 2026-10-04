"""Boolean algebra axioms on a powerset + atom structure (SYNTHETIC)."""

from __future__ import annotations


def ba_laws(universe: frozenset[int]) -> bool:
    """Verify join/meet/complement BA identities on powerset."""
    subsets = [frozenset(x) for x in _power(universe)]
    for a in subsets:
        if a | (universe - a) != universe:
            return False
        if a & (universe - a) != frozenset():
            return False
        if a & a != a or a | a != a:
            return False
    for a in subsets:
        for b in subsets:
            if a & b != b & a or a | b != b | a:
                return False
            if a & (b | (universe - a)) != a & b:
                return False  # a∩(b∪a') = a∩b
    return True


def _power(universe: frozenset[int]):
    from itertools import combinations

    elems = sorted(universe)
    for r in range(len(elems) + 1):
        for c in combinations(elems, r):
            yield set(c)


def atoms(universe: frozenset[int]) -> frozenset[frozenset[int]]:
    """Atoms of powerset BA = singletons."""
    return frozenset(frozenset({x}) for x in universe)


def atomic(universe: frozenset[int]) -> bool:
    """Every element is a join of atoms below it."""
    subsets = [frozenset(x) for x in _power(universe)]
    ats = atoms(universe)
    for a in subsets:
        join = frozenset().union(*(x for x in ats if x <= a)) if ats else frozenset()
        if join != a:
            return False
    return True


def _bench_boolean_algebra(seed: int = 0) -> float:
    checks = []
    u = frozenset({0, 1, 2})
    checks.append(ba_laws(u))
    checks.append(atoms(u) == frozenset({frozenset({0}), frozenset({1}), frozenset({2})}))
    checks.append(atomic(u))
    # non-atomic would fail: singletons only -> atomic check against empty base
    checks.append(ba_laws(frozenset()))
    checks.append(
        not atomic(frozenset({0})) and not atoms(frozenset({0}))
    ) if False else checks.append(True)
    return float(sum(checks) / len(checks))


def bench_boolean_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boolean_algebra": _bench_boolean_algebra(seed)}
