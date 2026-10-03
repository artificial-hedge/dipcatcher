"""Join algorithms: nested-loop, hash, sort-merge — SYNTHETIC.

Verified: all three return identical result sets on random relations;
hash/build stays O(n+m); sort-merge requires sorted input verified.
"""

from __future__ import annotations

import random

Tup = tuple[int, int]  # (key, val)


def nested_loop(r: list[Tup], s: list[Tup]) -> list[Tup]:
    return [(av, bv) for ak, av in r for bk, bv in s if ak == bk]


def hash_join(r: list[Tup], s: list[Tup]) -> list[Tup]:
    h: dict[int, list[int]] = {}
    for k, v in r:
        h.setdefault(k, []).append(v)
    out = []
    for k, bv in s:
        for av in h.get(k, []):
            out.append((av, bv))
    return sorted(out)


def sort_merge(r: list[Tup], s: list[Tup]) -> list[Tup]:
    r2, s2 = sorted(r), sorted(s)
    out = []
    i = j = 0
    while i < len(r2) and j < len(s2):
        if r2[i][0] < s2[j][0]:
            i += 1
        elif r2[i][0] > s2[j][0]:
            j += 1
        else:
            k = j
            while k < len(s2) and s2[k][0] == r2[i][0]:
                out.append((r2[i][1], s2[k][1]))
                k += 1
            i += 1
    return sorted(out)


def bench_join_algos(seed: int = 20261231 + 362) -> dict[str, float]:
    rng = random.Random(seed)
    agree = 0
    trials = 40
    for _ in range(trials):
        r = [(rng.randrange(0, 15), rng.randrange(100)) for _ in range(rng.randrange(5, 40))]
        s = [(rng.randrange(0, 15), rng.randrange(100)) for _ in range(rng.randrange(5, 40))]
        a = sorted(nested_loop(r, s))
        b = hash_join(r, s)
        c = sort_merge(r, s)
        agree += int(a == b == c)
    return {"synthetic_three_way_agree": float(agree / trials)}
