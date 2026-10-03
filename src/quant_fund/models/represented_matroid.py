"""Linear/represented matroids over finite fields (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def lin_indep_gf2(vectors: list[int], dim: int) -> set[frozenset[int]]:
    """Independence = linear independence of column vectors over GF(2),
    vectors as bitmask ints of length `dim`."""
    n = len(vectors)

    def independent(s: frozenset[int]) -> bool:
        # GF(2) rank via Gaussian elimination on bitmasks
        rows = [vectors[i] for i in s]
        basis: list[int] = []
        for r in rows:
            x = r
            for b in basis:
                x = min(x, x ^ b)
            if x:
                basis.append(x)
                basis.sort(reverse=True)
        return len(basis) == len(rows)

    return {
        frozenset(s)
        for r in range(n + 1)
        for s in combinations(range(n), r)
        if independent(frozenset(s))
    }


def _bench_represented_matroid(seed: int = 0) -> float:
    checks = []
    # Fano matroid = GF(2)^3 nonzero columns: 7 vectors
    fano_vecs = [1, 2, 3, 4, 5, 6, 7]  # nonzero elements of GF(2)^3
    fano = lin_indep_gf2(fano_vecs, 3)
    # every pair independent (distinct nonzero), triples iff sum != 0
    checks.append(all(frozenset(s) in fano for s in combinations(range(7), 2)))
    # {v1=1, v2=2, v1+v2=3}: 1^2^3 = 0 -> dependent
    checks.append(frozenset({0, 1, 2}) not in fano)
    # {1,2,4}: 1^2^4 = 7 != 0 -> independent
    checks.append(frozenset({0, 1, 3}) in fano)
    # Fano has 7 lines (dependent triples)
    dep3 = [s for s in combinations(range(7), 3) if frozenset(s) not in fano]
    checks.append(len(dep3) == 7)
    # rank 3: all 4-subsets dependent
    checks.append(all(frozenset(s) not in fano for s in combinations(range(7), 4)))
    return float(sum(checks) / len(checks))


def bench_represented_matroid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_represented_matroid": _bench_represented_matroid(seed)}
