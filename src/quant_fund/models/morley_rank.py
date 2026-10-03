"""Morley rank on finite Boolean algebras of definable sets (SYNTHETIC)."""

from __future__ import annotations


def cantor_bendixson_rank(
    sets: list[frozenset[int]], universe: frozenset[int]
) -> dict[frozenset[int], int]:
    """Rank of a definable set = CB-rank in the Stone space of types.

    Toy semantics for a strongly minimal structure: the definable sets
    are the finite/cofinite Boolean algebra, where "small" (rank 0) means
    strictly smaller than its complement and "large" (rank 1) means the
    complement is strictly smaller — the finite-model analogue of
    finite-vs-cofinite.
    """
    rank: dict[frozenset[int], int] = {}
    for s in sets:
        rank[s] = 1 if len(universe - s) < len(s) else 0
    return rank


def morley_degree(components: list[frozenset[int]]) -> int:
    """Degree = number of disjoint full-rank components partitioning U."""
    return len(components)


def _bench_morley_rank(seed: int = 0) -> float:
    checks = []
    u = frozenset(range(10))
    # in the pure set (strongly minimal): finite -> rank 0, cofinite -> 1
    sets = [frozenset({0}), frozenset(range(5)), u, frozenset(range(1, 10))]
    r = cantor_bendixson_rank(sets, u)
    checks.append(r[frozenset({0})] == 0)
    checks.append(r[frozenset(range(5))] == 0)  # 5 not < 5 -> small
    checks.append(r[u] == 1)
    # cofinite subset also rank 1 (complement finite)
    checks.append(r[frozenset(range(1, 10))] == 1)
    # degree of a 2-piece partition is 2
    parts = [frozenset(range(5)), frozenset(range(5, 10))]
    checks.append(morley_degree(parts) == 2)
    # union of finite sets stays rank 0
    checks.append(max(r[s] for s in sets[:2]) == 0)
    return float(sum(checks) / len(checks))


def bench_morley_rank(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morley_rank": _bench_morley_rank(seed)}
