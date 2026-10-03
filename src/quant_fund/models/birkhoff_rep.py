"""Birkhoff representation: distributive lattice = J(P) (SYNTHETIC)."""

from __future__ import annotations


def join_irreducibles(downsets: set[frozenset[int]]) -> set[frozenset[int]]:
    """Join-irreducibles of J(P): downsets that are not a union of two
    strictly smaller downsets (equivalently principal ideals in P)."""
    out = set()
    for d in downsets:
        if not d:
            continue
        reducible = False
        for a in downsets:
            for b in downsets:
                if a < d and b < d and (a | b) == d:
                    reducible = True
        if not reducible:
            out.add(d)
    return out


def _bench_birkhoff_rep(seed: int = 0) -> float:
    import itertools

    checks = []
    # J(chain-2) = chain-3 lattice: join-irreducibles are {0} and {0,1}
    # all downsets of chain-2 = {},{0},{0,1}
    ds = {frozenset(), frozenset({0}), frozenset({0, 1})}
    ji = join_irreducibles(ds)
    checks.append(ji == {frozenset({0}), frozenset({0, 1})})
    # boolean B2 J(P)=powerset: join-irreducibles are the atoms {0},{1}
    ds2 = {frozenset(c) for r in range(3) for c in itertools.combinations(range(2), r)}
    ji2 = join_irreducibles(ds2)
    checks.append(ji2 == {frozenset({0}), frozenset({1})})
    # bottom is never join-irreducible
    checks.append(frozenset() not in ji)
    # every downset is a union of join-irreducibles contained in it
    checks.append(
        all(d == frozenset().union(*[j for j in ji if j <= d]) if d else True for d in ds)
    )
    return float(sum(checks) / len(checks))


def bench_birkhoff_rep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_birkhoff_rep": _bench_birkhoff_rep(seed)}
