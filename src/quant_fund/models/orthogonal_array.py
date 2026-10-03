"""Orthogonal arrays OA(N, k, s, t) (SYNTHETIC)."""

from __future__ import annotations

import itertools


def is_oa(runs: list[tuple[int, ...]], s: int, t: int) -> bool:
    """Check the OA property: in every t-column subset each t-tuple of
    symbols appears exactly N/s^t times."""
    if not runs:
        return False
    n = len(runs)
    k = len(runs[0])
    if n % (s**t) != 0:
        return False
    lam = n // (s**t)
    for cols in itertools.combinations(range(k), t):
        counts: dict[tuple[int, ...], int] = {}
        for r in runs:
            key = tuple(r[c] for c in cols)
            counts[key] = counts.get(key, 0) + 1
        if len(counts) != s**t or any(v != lam for v in counts.values()):
            return False
    return True


def _bench_orthogonal_array(seed: int = 0) -> float:
    checks = []
    # OA(4,3,2,2): the 4 binary triples forming a 2-universal set
    runs: list[tuple[int, ...]] = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    checks.append(is_oa(runs, 2, 2))
    # repetition of a single run is not OA
    rep: list[tuple[int, ...]] = [(0, 0, 0)] * 4
    checks.append(not is_oa(rep, 2, 2))
    # all 8 binary triples form OA(8,3,2,2) with lambda 2
    all3: list[tuple[int, ...]] = [
        tuple(int(b) for b in format(i, "03b")) for i in range(8)
    ]
    checks.append(is_oa(all3, 2, 2))
    # dropping one run breaks balance
    checks.append(not is_oa(all3[:-1], 2, 2))
    # OA(9,4,3,2) = affine plane rows: (x, y, x+y, x+2y) over F_3
    runs3: list[tuple[int, ...]] = [
        (x, y, (x + y) % 3, (x + 2 * y) % 3) for x in range(3) for y in range(3)
    ]
    checks.append(is_oa(runs3, 3, 2))
    return float(sum(checks) / len(checks))


def bench_orthogonal_array(seed: int = 0) -> dict[str, float]:
    return {"synthetic_orthogonal_array": _bench_orthogonal_array(seed)}
