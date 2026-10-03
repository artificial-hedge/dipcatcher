"""Ryll-Nardzewski: orbit counts under automorphisms (SYNTHETIC)."""

from __future__ import annotations

from itertools import permutations


def automorphisms(n: int, edges: set[tuple[int, int]]) -> list[list[int]]:
    """All permutations of the vertex set preserving the edge relation."""
    und = {frozenset(e) for e in edges}
    autos = []
    for p in permutations(range(n)):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if (frozenset({i, j}) in und) != (frozenset({p[i], p[j]}) in und):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            autos.append(list(p))
    return autos


def orbits_of_tuples(autos: list[list[int]], n: int, k: int) -> int:
    """Number of orbits of ordered k-tuples under the automorphism group."""
    from itertools import product

    tuples = list(product(range(n), repeat=k))
    seen: set[tuple[int, ...]] = set()
    count = 0
    for t in tuples:
        if t in seen:
            continue
        count += 1
        for a in autos:
            seen.add(tuple(a[v] for v in t))
    return count


def _bench_omega_categoricity(seed: int = 0) -> float:
    checks = []
    # complete graph K4: automorphism group S4 -> 1 orbit of vertices,
    # 2 orbits of ordered pairs (diagonal, off-diagonal), 3 of triples
    # (all-equal, two-equal, all-distinct)
    k4 = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    autos = automorphisms(4, k4)
    checks.append(len(autos) == 24)
    checks.append(orbits_of_tuples(autos, 4, 1) == 1)
    checks.append(orbits_of_tuples(autos, 4, 2) == 2)
    checks.append(orbits_of_tuples(autos, 4, 3) == 5)  # Bell(3) equality patterns
    # 4-tuples under S4: Bell(4) = 15 equality patterns
    checks.append(orbits_of_tuples(autos, 4, 4) == 15)
    # path graph P4 (0-1-2-3): automorphism = {id, reflection} -> 2 vertex orbits
    p4 = {(0, 1), (1, 2), (2, 3)}
    autos2 = automorphisms(4, p4)
    checks.append(len(autos2) == 2)
    checks.append(orbits_of_tuples(autos2, 4, 1) == 2)
    return float(sum(checks) / len(checks))


def bench_omega_categoricity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_omega_categoricity": _bench_omega_categoricity(seed)}
