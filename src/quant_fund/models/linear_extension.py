"""Count of linear extensions of a poset (SYNTHETIC)."""

from __future__ import annotations

import itertools


def count_ext(elems: tuple[int, ...], leq) -> int:
    """Linear extensions = topological orderings; brute force for small
    posets."""
    n = 0
    for perm in itertools.permutations(elems):
        ok = True
        for i, a in enumerate(perm):
            for b in perm[i + 1 :]:
                if leq(b, a) and a != b:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            n += 1
    return n


def _bench_linear_extension(seed: int = 0) -> float:
    checks = []
    # chain has exactly 1 linear extension
    checks.append(count_ext(tuple(range(3)), lambda a, b: a <= b) == 1)
    # antichain-3 has 3! = 6
    checks.append(count_ext(tuple(range(3)), lambda a, b: a == b) == 6)
    # V-poset (a,b < c): 2 extensions (a,b,c) and (b,a,c)
    checks.append(
        count_ext(
            tuple(range(3)),
            lambda a, b: a == b or (a in (0, 1) and b == 2),
        )
        == 2
    )
    # diamond a < b,c < d: 2 extensions
    checks.append(
        count_ext(
            tuple(range(4)),
            lambda a, b: a == b or a == 0 and b in (1, 2, 3) or a in (1, 2) and b == 3,
        )
        == 2
    )
    return float(sum(checks) / len(checks))


def bench_linear_extension(seed: int = 0) -> dict[str, float]:
    return {"synthetic_linear_extension": _bench_linear_extension(seed)}
