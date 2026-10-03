"""Krull dimension via prime chains (SYNTHETIC)."""

from __future__ import annotations


def dim_polynomial_ring(n_vars: int) -> int:
    """dim k[x_1..x_n] = n."""
    return n_vars


def longest_chain(primes: list[tuple[int, ...]]) -> int:
    """Length of the longest chain in a poset of prime ideals
    (elements = subsets represented as tuples of generator indices)."""
    best = 0
    for p in primes:
        depth = 1
        frontier = [p]
        while frontier:
            nxt = [q for q in primes for f in frontier if set(f) < set(q)]
            if nxt:
                depth += 1
                frontier = nxt[:1]
            else:
                frontier = []
        best = max(best, depth)
    return best


def _bench_krull_dim(seed: int = 0) -> float:
    checks = []
    checks.append(dim_polynomial_ring(3) == 3)
    checks.append(dim_polynomial_ring(0) == 0)  # k: dim 0
    # Z: (0) < (p) -> dim 1
    checks.append(longest_chain([tuple(), (2,)]) == 2)
    # Z/p is a field: dim 0, single prime
    checks.append(longest_chain([tuple()]) == 1)
    # k[x,y]: (0) < (x) < (x,y) -> chain of 3 = dim 2
    checks.append(longest_chain([tuple(), (0,), (0, 1)]) == 3)
    # dim R = sup chain - 1: 3-element chain -> dim 2
    checks.append(longest_chain([tuple(), (0,), (0, 1)]) - 1 == 2)
    return float(sum(checks) / len(checks))


def bench_krull_dim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_krull_dim": _bench_krull_dim(seed)}
