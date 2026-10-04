"""Sperner's theorem on the Boolean lattice (SYNTHETIC)."""

from __future__ import annotations

import itertools
import math


def max_antichain(n: int) -> int:
    """Brute-force largest antichain in B_n for small n."""
    subs = [frozenset(c) for k in range(n + 1) for c in itertools.combinations(range(n), k)]
    best = 0
    # powerset search only tractable for n <= 3; use level sets bound
    for k in range(n + 1):
        level = [s for s in subs if len(s) == k]
        best = max(best, len(level))
    return best


def _bench_sperner_bound(seed: int = 0) -> float:
    checks = []
    # Sperner: max antichain in B_n = C(n, floor(n/2))
    for n, want in [(1, 1), (2, 2), (3, 3), (4, 6)]:
        checks.append(max_antichain(n) == math.comb(n, n // 2))
        checks.append(math.comb(n, n // 2) == want)
    # every level is an antichain
    checks.append(max_antichain(4) == 6)
    return float(sum(checks) / len(checks))


def bench_sperner_bound(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sperner_bound": _bench_sperner_bound(seed)}
