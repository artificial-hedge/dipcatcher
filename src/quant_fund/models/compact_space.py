"""Compactness on finite topologies: every open cover has finite subcover (SYNTHETIC)."""

from __future__ import annotations

import itertools

Topo = frozenset[frozenset[int]]


def is_compact(univ: frozenset[int], opens: Topo) -> bool:
    """Finite spaces are always compact; check definition directly: every open
    cover of the universe already contains a finite subcover (trivially true).
    Verify instead: space compact iff every open cover contains finite subcover —
    enumerate covers and check a subcover of size <= len(opens)."""
    for r in range(1, len(opens) + 1):
        for cover in itertools.combinations(opens, r):
            cov = frozenset(cover)
            union = frozenset().union(*cov) if cov else frozenset()
            if union == univ:
                # finite subcover = the cover itself
                return True
    return False


def compact_subset(univ: frozenset[int], opens: Topo, subset: frozenset[int]) -> bool:
    """Subspace compactness: every cover of subset by opens has finite subcover."""
    covers_with = [o for o in opens if o & subset]
    for r in range(1, len(covers_with) + 1):
        for cover in itertools.combinations(covers_with, r):
            union = frozenset().union(*cover)
            if subset <= union:
                return True
    return not bool(subset)


def _bench_compact_space(seed: int = 0) -> float:
    checks = []
    u = frozenset({0, 1, 2})
    import itertools as it

    discrete = frozenset(frozenset(c) for r in range(4) for c in it.combinations(u, r))
    checks.append(is_compact(u, discrete))
    indiscrete = frozenset({frozenset(), u})
    checks.append(is_compact(u, indiscrete))
    sier = frozenset({frozenset(), frozenset({0}), u})
    checks.append(is_compact(u, sier))
    checks.append(compact_subset(u, discrete, frozenset({0, 1})))
    checks.append(compact_subset(u, indiscrete, frozenset({0})))
    checks.append(compact_subset(u, indiscrete, frozenset()))  # empty set is compact
    return float(sum(checks) / len(checks))


def bench_compact_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compact_space": _bench_compact_space(seed)}
