"""Algebraic closure acl(A) via automorphism orbits (SYNTHETIC)."""

from __future__ import annotations

from itertools import permutations


def _autos_fixing(n: int, edges: set[tuple[int, int]], fixed: set[int]) -> list[list[int]]:
    und = {frozenset(e) for e in edges}
    out = []
    for p in permutations(range(n)):
        if any(p[v] != v for v in fixed):
            continue
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if (frozenset({i, j}) in und) != (frozenset({p[i], p[j]}) in und):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(list(p))
    return out


def acl(n: int, edges: set[tuple[int, int]], a_set: set[int]) -> set[int]:
    """Elements whose orbit under Aut(M/A) is a singleton are algebraic:
    in a finite structure acl(A) = elements fixed by all A-automorphisms
    extended to the pointwise stabilizer."""
    autos = _autos_fixing(n, edges, a_set)
    out = set(a_set)
    for v in range(n):
        orbit = {a[v] for a in autos}
        if len(orbit) == 1:
            out.add(v)
    return out


def _bench_acl_closure(seed: int = 0) -> float:
    checks = []
    # complete graph K4: acl(empty) = empty (all elements symmetric)
    k4 = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    checks.append(acl(4, k4, set()) == set())
    checks.append(acl(4, k4, {0}) == {0})  # fixing 0 leaves others symmetric
    # path P4: fixing vertex 0 fixes everything via reflection constraint
    p4 = {(0, 1), (1, 2), (2, 3)}
    checks.append(acl(4, p4, {0}) == {0, 1, 2, 3})  # rigid after anchor
    # star graph: center 0, leaves 1..4. acl({center}) = {center} (leaves symmetric)
    star = {(0, i) for i in range(1, 5)}
    checks.append(acl(5, star, {0}) == {0})
    # acl({leaf 1}) = {0,1}: fixing leaf fixes center, other leaves free
    checks.append(acl(5, star, {1}) == {0, 1})
    # acl is a closure operator: A subset acl(A)
    checks.append({2} <= acl(4, k4, {2}))
    return float(sum(checks) / len(checks))


def bench_acl_closure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_acl_closure": _bench_acl_closure(seed)}
