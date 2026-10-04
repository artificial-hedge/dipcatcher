"""Permutation representations: Cayley embedding and sign (SYNTHETIC)."""

from __future__ import annotations

import itertools


def cayley_perm(g: int, table: list[list[int]]) -> tuple[int, ...]:
    """Cayley action of element g on the group: permutation of indices."""
    return tuple(table[g][h] for h in range(len(table)))


def sign_parity(perm: tuple[int, ...]) -> int:
    """Sign of a permutation: +1 even, -1 odd."""
    seen = [False] * len(perm)
    sign = 1
    for i in range(len(perm)):
        if seen[i]:
            continue
        j = i
        length = 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length and length % 2 == 0:
            sign = -sign
    return sign


def faithful(cayley: dict[int, tuple[int, ...]]) -> bool:
    """The Cayley embedding is faithful: distinct elements give distinct
    permutations."""
    return len(set(cayley.values())) == len(cayley)


def _bench_permutation_poly(seed: int = 0) -> float:
    checks = []
    # C3 multiplication table
    tab3 = [[(i * j) % 3 for j in range(3)] for i in range(3)]
    c = {g: cayley_perm(g, tab3) for g in range(3)}
    checks.append(faithful(c))
    # generator 1 maps to the 3-cycle (1,2,0) -> even? a 3-cycle is even
    checks.append(sign_parity(c[1]) == 1)
    # identity maps to identity permutation: even
    checks.append(sign_parity(c[0]) == 1)
    # S3 table on a small model: use C2 x C2 (V4): every nonidentity
    # element is a product of two disjoint transpositions -> even
    tabv = [[i ^ j for j in range(4)] for i in range(4)]
    cv = {g: cayley_perm(g, tabv) for g in range(4)}
    checks.append(faithful(cv))
    checks.append(sign_parity(cv[1]) == 1)
    # a transposition is odd
    checks.append(sign_parity((1, 0)) == -1)
    # 4-cycle is odd: sign = -1
    checks.append(sign_parity((1, 2, 3, 0)) == -1)
    # sanity: itertools permutations count 3! = 6
    checks.append(len(list(itertools.permutations(range(3)))) == 6)
    return float(sum(checks) / len(checks))


def bench_permutation_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_permutation_poly": _bench_permutation_poly(seed)}
