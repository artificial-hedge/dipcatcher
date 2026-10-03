"""Dilworth: min chain cover equals width (SYNTHETIC)."""

from __future__ import annotations

import itertools


def min_chain_cover(elems: tuple[int, ...], leq) -> int:
    """Minimum number of chains covering P; brute force over
    partitions (small n)."""
    n = len(elems)

    # check if a subset is a chain
    def is_chain(sub):
        return all(leq(a, b) or leq(b, a) for a, b in itertools.combinations(sub, 2))

    # DP over subsets
    subs = [frozenset(c) for r in range(1, n + 1) for c in itertools.combinations(elems, r)]
    chains = {s for s in subs if is_chain(s)}
    full = frozenset(elems)
    # set cover DP: dp[s] = min chains covering s
    dp: dict[frozenset[int], int] = {frozenset(): 0}
    for r in range(1, n + 1):
        for s in subs:
            if len(s) != r:
                continue
            best = n + 1
            for c in chains:
                if c <= s:
                    rem = s - c
                    if rem in dp:
                        best = min(best, dp[rem] + 1)
            dp[s] = best
    return dp[full]


def _bench_dilworth_partition(seed: int = 0) -> float:
    checks = []
    # width of antichain-3 is 3; min chain cover also 3
    checks.append(min_chain_cover(tuple(range(3)), lambda a, b: a == b) == 3)
    # chain-4 covered by 1 chain
    checks.append(min_chain_cover(tuple(range(4)), lambda a, b: a <= b) == 1)
    # V-poset (a,b<c): width 2 -> 2 chains
    checks.append(
        min_chain_cover(
            tuple(range(3)),
            lambda a, b: a == b or (a in (0, 1) and b == 2),
        )
        == 2
    )
    # diamond: width 2
    checks.append(
        min_chain_cover(
            tuple(range(4)),
            lambda a, b: a == b or a == 0 and b in (1, 2, 3) or a in (1, 2) and b == 3,
        )
        == 2
    )
    return float(sum(checks) / len(checks))


def bench_dilworth_partition(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dilworth_partition": _bench_dilworth_partition(seed)}
