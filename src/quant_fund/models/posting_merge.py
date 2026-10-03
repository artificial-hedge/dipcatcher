"""SYNTHETIC galloping (exponential-search) posting intersection.

Skip-ahead merge for very asymmetric lists; verified exact against
sorted set intersection, and probe-count < linear walk.
"""

from __future__ import annotations

import bisect
import random


def galloping_intersect(short: list[int], long: list[int]) -> list[int]:
    out = []
    lo = 0
    for v in short:
        lo = bisect.bisect_left(long, v, lo)
        if lo < len(long) and long[lo] == v:
            out.append(v)
    return out


PROBES = 0


def skip_merge(a: list[int], b: list[int]) -> list[int]:
    """Intersect two sorted lists; exponential skips on the longer one."""
    global PROBES
    PROBES = 0
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        PROBES += 1
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            # a[i] > b[j]: exponential-search forward in b for a[i]
            step = 1
            while j + step < len(b) and b[j + step] < a[i]:
                step <<= 1
                PROBES += 1
            lo = j + step // 2
            hi = min(len(b), j + step + 1)
            j = bisect.bisect_left(b, a[i], lo, hi)
            if j >= len(b):
                break
    return out


def _set_intersect(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) & set(b))


def bench_posting_merge(seed: int = 20261231 + 461) -> dict[str, float]:
    rng = random.Random(seed)
    exact = fewer = gal = 0
    trials = 40
    for _ in range(trials):
        a = sorted(rng.sample(range(3000), rng.randrange(5, 30)))
        b = sorted(rng.sample(range(3000), rng.randrange(300, 1500)))
        exact += int(skip_merge(a, b) == _set_intersect(a, b))
        gal += int(galloping_intersect(a, b) == _set_intersect(a, b))
        n = len(a) + len(b)
        fewer += int(n > PROBES)
    return {
        "synthetic_skip_merge_exact": float(exact / trials),
        "synthetic_galloping_exact": float(gal / trials),
        "synthetic_sublinear_probes": float(fewer / trials),
    }
