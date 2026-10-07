"""Permutation-group arithmetic (wave 281) (SYNTHETIC).

Cycle decomposition, order (lcm of cycle lengths), sign (parity), and
composition — verified against brute-force oracle properties.
"""

import numpy as np

_SEED = 20261231 + 771


def cycles(p: list[int]) -> list[list[int]]:
    seen = [False] * len(p)
    out = []
    for i in range(len(p)):
        if seen[i]:
            continue
        cyc, j = [], i
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = p[j]
        out.append(cyc)
    return out


def order(p: list[int]) -> int:
    from math import gcd

    o = 1
    for c in cycles(p):
        if len(c) > 1:
            o = o * len(c) // gcd(o, len(c))
    return o


def sign(p: list[int]) -> int:
    # sign = product over cycles of (-1)^(len-1); also = (-1)^(n - #cycles)
    n_cyc = sum(1 for c in cycles(p) if len(c) > 0)
    return -1 if (len(p) - n_cyc) % 2 else 1


def compose(p: list[int], q: list[int]) -> list[int]:
    return [p[q[i]] for i in range(len(p))]


def bench_perm_group(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(6):
        p = list(rng.permutation(7))
        # order: smallest k with p^k = id (brute force)
        cur, k = p[:], 1
        while cur != list(range(7)):
            cur = compose(cur, p)
            k += 1
            if k > 5040:
                break
        ok += int(order(p) == k)
        # sign vs inversion-parity oracle
        inv = sum(1 for i in range(7) for j in range(i + 1, 7) if p[i] > p[j])
        ok += int(sign(p) == (-1 if inv % 2 else 1))
    return {"synthetic_perm_arith": float(ok == 12)}
