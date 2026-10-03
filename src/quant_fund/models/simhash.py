"""SimHash locality-sensitive fingerprint (synthetic).

64-bit fingerprint: bit i = sign(Σ_features w·h_i(feature)).
Verified: near-duplicates (few edits) have small Hamming distance;
unrelated docs have ~32-bit distance; distance is symmetric.
"""

from __future__ import annotations

import hashlib
import random

BITS = 64


def _h(feat: str, i: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{i}:{feat}".encode()).digest()[:8], "little") & 1


def simhash(feats: dict[str, float]) -> int:
    acc = [0.0] * BITS
    for f, w in feats.items():
        for i in range(BITS):
            acc[i] += w if _h(f, i) else -w
    out = 0
    for i, a in enumerate(acc):
        if a > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def bench_simhash(seed: int = 20261231 + 315) -> dict[str, float]:
    rng = random.Random(seed)
    near_ok = far_ok = sym = 0
    trials = 40
    for _ in range(trials):
        base = {f"f{i}": rng.random() + 0.5 for i in range(40)}
        # near dup: drop/perturb ~10%
        dup = {f: w * rng.uniform(0.95, 1.05) for f, w in base.items()}
        for k in rng.sample(sorted(dup), 4):
            del dup[k]
        far = {f"g{i}": rng.random() for i in range(40)}
        h1, h2, h3 = simhash(base), simhash(dup), simhash(far)
        near_ok += int(hamming(h1, h2) <= 16)
        far_ok += int(hamming(h1, h3) >= 16)
        sym += int(hamming(h1, h2) == hamming(h2, h1))
    return {
        "synthetic_near_detects": float(near_ok / trials),
        "synthetic_far_separates": float(far_ok / trials),
        "synthetic_symmetric": float(sym / trials),
    }
