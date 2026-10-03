"""Lattice of downsets J(P) (SYNTHETIC)."""

from __future__ import annotations

import itertools


def downsets(elems: tuple[int, ...], leq) -> set[frozenset[int]]:
    """All down-sets of the poset (elems, leq)."""
    out: set[frozenset[int]] = set()
    for r in range(len(elems) + 1):
        for sub in itertools.combinations(elems, r):
            s = frozenset(sub)
            if all(not leq(y, x) or y in s for x in s for y in elems):
                out.add(s)
    return out


def _bench_downset_lattice(seed: int = 0) -> float:
    checks = []

    def leq(a: int, b: int) -> bool:
        return a <= b

    chain3 = tuple(range(3))
    ds = downsets(chain3, leq)
    # J(chain-3) is the chain lattice with 4 elements
    checks.append(len(ds) == 4)
    # antichain-2 downsets = full powerset (4 sets)
    anti = tuple(range(2))
    ds2 = downsets(anti, lambda a, b: a == b)
    checks.append(len(ds2) == 4)

    # diamond poset (V shape): 3 minimal-downsets + 5 total
    def leq_v(a: int, b: int) -> bool:
        return a == b or (a in (0, 1) and b == 2)

    v = tuple(range(3))
    ds3 = downsets(v, leq_v)
    checks.append(len(ds3) == 5)
    # union and intersection of downsets are downsets (closed under
    # lattice ops)
    a = frozenset({0})
    b2 = frozenset({1})
    checks.append((a | b2) in ds3)
    checks.append((a & b2) in ds3)
    return float(sum(checks) / len(checks))


def bench_downset_lattice(seed: int = 0) -> dict[str, float]:
    return {"synthetic_downset_lattice": _bench_downset_lattice(seed)}
