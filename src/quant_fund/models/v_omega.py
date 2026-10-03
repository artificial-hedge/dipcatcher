"""Hereditarily finite sets V_omega: ZFC axiom witnesses on finite levels (SYNTHETIC)."""

from __future__ import annotations

import itertools

HF = frozenset


def v_level(n: int) -> frozenset:
    v: frozenset = frozenset()
    for _ in range(n):
        v = power(v)
    return v


def power(s: frozenset) -> frozenset:
    items = list(s)
    return frozenset(
        frozenset(c) for r in range(len(items) + 1) for c in itertools.combinations(items, r)
    )


def is_ordinal_hf(s: frozenset) -> bool:
    """HF ordinal = von Neumann ordinal: transitive & well-ordered by membership."""
    if not all(is_ordinal_hf(m) for m in s):
        return False
    # transitivity: every member's members are members
    for m in s:
        for x in m:
            if x not in s:
                return False
    # trichotomy on members: for distinct a,b: a in b or b in a
    for a in s:
        for b in s:
            if a != b and a not in b and b not in a:
                return False
    return True


def pair(a: frozenset, b: frozenset) -> frozenset:
    return frozenset({a, b})


def union_of(s: frozenset) -> frozenset:
    out: frozenset = frozenset()
    for m in s:
        out = out | m
    return out


def successor(s: frozenset) -> frozenset:
    return s | frozenset({s})


def _bench_v_omega(seed: int = 0) -> float:
    checks = []
    v0, v1, v2, v3 = (v_level(i) for i in range(4))
    checks.append(v0 == frozenset())
    checks.append(v1 == frozenset({v0}))
    checks.append(len(v3) == 4)
    # von Neumann ordinals 0,1,2,3 are elements of V_4
    checks.append(is_ordinal_hf(v0) and is_ordinal_hf(v1) and is_ordinal_hf(v2))
    checks.append(
        not is_ordinal_hf(frozenset({v0, v2}))
    )  # {0,2} not transitive (1 in 2, 1 notin set)
    # pair/union axioms
    checks.append(pair(v0, v1) == frozenset({v0, v1}))
    checks.append(union_of(pair(v0, v1)) == v1)
    checks.append(successor(v1) == v2)
    return float(sum(checks) / len(checks))


def bench_v_omega(seed: int = 0) -> dict[str, float]:
    return {"synthetic_v_omega": _bench_v_omega(seed)}
