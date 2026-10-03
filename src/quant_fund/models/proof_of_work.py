"""Proof-of-work mining — SYNTHETIC sha256d vs difficulty target.

Verified: found nonce satisfies target; expected-work metric ≈
geometric mean of trials.
"""

from __future__ import annotations

import hashlib
import random


def _h2(b: bytes) -> int:
    return int.from_bytes(hashlib.sha256(hashlib.sha256(b).digest()).digest(), "big")


def mine(header: bytes, bits: int, nonce0: int = 0, max_iter: int = 1 << 26) -> tuple[int, int]:
    """Returns (nonce, hashes_tried). Target = 2^(256-bits)."""
    target = 1 << (256 - bits)
    n = nonce0
    while n - nonce0 < max_iter:
        if _h2(header + n.to_bytes(8, "little")) < target:
            return n, n - nonce0 + 1
        n += 1
    raise ValueError("exhausted")


def bench_proof_of_work(seed: int = 20261231 + 341) -> dict[str, float]:
    rng = random.Random(seed)
    bits = 12  # expect ~4096 hashes
    valid = 0
    total_work = 0
    trials = 20
    for _ in range(trials):
        h = rng.randbytes(76)
        nonce, work = mine(h, bits)
        target = 1 << (256 - bits)
        valid += int(_h2(h + nonce.to_bytes(8, "little")) < target)
        total_work += work
    exp = float(1 << bits)
    ratio = (total_work / trials) / exp
    return {
        "synthetic_valid_solutions": float(valid / trials),
        "synthetic_mean_work_ratio": float(ratio),
        "synthetic_work_near_geometric": float(0.25 <= ratio <= 4.0),
    }
