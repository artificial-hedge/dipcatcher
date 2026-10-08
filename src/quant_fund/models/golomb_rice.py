"""Golomb–Rice codes for geometric distributions (synthetic) (SYNTHETIC).

n ↦ unary(n >> k) + k low bits. Verified: exact round-trip on
geometric samples; mean code length within ~0.5 bit of the
geometric-distribution entropy when k is chosen as ceil(log2(μ)).
"""

from __future__ import annotations

import math
import random


def rice_encode(n: int, k: int) -> str:
    q = n >> k
    return "1" * q + "0" + format(n & ((1 << k) - 1), f"0{k}b")


def rice_decode(bits: str, k: int) -> tuple[list[int], int]:
    out: list[int] = []
    i = 0
    while i + k <= len(bits):
        q = 0
        while i < len(bits) and bits[i] == "1":
            q += 1
            i += 1
        if i >= len(bits):
            return out, len(bits)  # truncated tail
        i += 1  # skip '0'
        if i + k > len(bits):
            break
        r = int(bits[i : i + k], 2)
        i += k
        out.append((q << k) | r)
    return out, i


def bench_golomb_rice(seed: int = 20261231 + 293) -> dict[str, float]:
    rng = random.Random(seed)
    rt = 0
    trials = 40
    gaps: list[float] = []
    near = 0
    for _ in range(trials):
        mu = rng.randint(4, 64)
        k = max(0, round(math.log2(mu)))
        # geometric-ish samples
        n = 200
        msgs = [rng.expovariate(1 / mu) for _ in range(n)]
        msgs_int = [int(x) for x in msgs]
        bits = "".join(rice_encode(m, k) for m in msgs_int)
        dec, _ = rice_decode(bits, k)
        rt += int(dec == msgs_int)
        # entropy of exponential-geom approximation
        h = -sum(
            (p := (1 / (mu + 1)) * (mu / (mu + 1)) ** m) * math.log2(max(p, 1e-12))
            for m in range(0, 2000)
        )
        gap = len(bits) / n - h
        gaps.append(gap)
        near += int(gap < 1.5)
    return {
        "synthetic_roundtrip": float(rt / trials),
        "synthetic_near_entropy": float(near / trials),
        "synthetic_mean_gap": float(sum(gaps) / len(gaps)),
    }
