"""Normality on finite spaces: disjoint closed sets separable by opens (SYNTHETIC)."""

from __future__ import annotations

import itertools

Topo = frozenset[frozenset[int]]


def closed_sets(univ: frozenset[int], opens: Topo) -> Topo:
    return frozenset(univ - o for o in opens)


def separated(univ: frozenset[int], opens: Topo, a: frozenset[int], b: frozenset[int]) -> bool:
    """Exist disjoint opens U sup a, V sup b."""
    return any(a <= u and b <= v and not (u & v) for u, v in itertools.product(opens, repeat=2))


def is_normal(univ: frozenset[int], opens: Topo) -> bool:
    """Every pair of disjoint closed sets can be separated."""
    cls = closed_sets(univ, opens)
    for a, b in itertools.product(cls, repeat=2):
        if not (a & b) and not separated(univ, opens, a, b):
            return False
    return True


def _bench_tietze_urysohn(seed: int = 0) -> float:
    checks = []
    u = frozenset({0, 1})
    disc = frozenset(frozenset(c) for r in range(3) for c in itertools.combinations(u, r))
    ind = frozenset({frozenset(), u})
    checks.append(is_normal(u, disc))
    checks.append(is_normal(u, ind))
    checks.append(separated(u, disc, frozenset({0}), frozenset({1})))
    checks.append(not separated(u, ind, frozenset({0}), frozenset({1})))
    cls = closed_sets(u, disc)
    checks.append(cls == disc)
    # Sierpinski: closed sets {empty, {1}, {0,1}} — {1} vs {} only; normal
    sier = frozenset({frozenset(), frozenset({0}), u})
    checks.append(is_normal(u, sier))
    return float(sum(checks) / len(checks))


def bench_tietze_urysohn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tietze_urysohn": _bench_tietze_urysohn(seed)}
