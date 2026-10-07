"""Bloom filter with optimal m,k sizing (synthetic) (SYNTHETIC).

m = −n·ln(p)/(ln 2)², k = round(m/n·ln 2) hash functions via
double hashing h_i = h1 + i·h2. Verified: zero false negatives;
observed FPR within slack of the design bound; occupancy ≈
theoretical (1−e^{−kn/m}).
"""

from __future__ import annotations

import hashlib
import math
import random


def _h(x: int, i: int, m: int) -> int:
    b = hashlib.sha256(f"{x}".encode()).digest()
    h1 = int.from_bytes(b[:8], "little")
    h2 = int.from_bytes(b[8:16], "little") | 1
    return (h1 + i * h2) % m


class Bloom:
    def __init__(self, n: int, p: float) -> None:
        self.m = max(8, int(-n * math.log(p) / (math.log(2) ** 2)))
        self.k = max(1, round(self.m / n * math.log(2)))
        self.bits = bytearray(self.m)

    def add(self, x: int) -> None:
        for i in range(self.k):
            self.bits[_h(x, i, self.m)] = 1

    def contains(self, x: int) -> bool:
        return all(self.bits[_h(x, i, self.m)] for i in range(self.k))


def bench_bloom_filter(seed: int = 20261231 + 310) -> dict[str, float]:
    rng = random.Random(seed)
    fn = within = 0
    trials = 30
    fprs: list[float] = []
    occ_errs: list[float] = []
    for _ in range(trials):
        n = rng.randint(50, 400)
        p = rng.choice([0.01, 0.05])
        bf = Bloom(n, p)
        items = [rng.randrange(10**9) for _ in range(n)]
        for x in items:
            bf.add(x)
        fn += int(all(bf.contains(x) for x in items))
        neg = [rng.randrange(10**9, 10**10) for _ in range(n * 2)]
        fpr = sum(1 for x in neg if bf.contains(x)) / len(neg)
        fprs.append(fpr)
        within += int(fpr <= p * 3 + 0.01)
        occ = sum(bf.bits) / bf.m
        occ_errs.append(abs(occ - (1 - math.exp(-bf.k * n / bf.m))))
    return {
        "synthetic_no_fn": float(fn / trials),
        "synthetic_fpr_within": float(within / trials),
        "synthetic_mean_fpr": float(sum(fprs) / len(fprs)),
        "synthetic_occ_err": float(sum(occ_errs) / len(occ_errs)),
    }
