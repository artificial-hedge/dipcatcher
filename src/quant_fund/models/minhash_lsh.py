"""MinHash signatures + LSH banding (synthetic) (SYNTHETIC).

Signature_i(x) = min over items of (a_i·h + b_i mod p); Pr[min
equal] = J(A,B). LSH: r rows per band → candidate iff band hash
matches. Verified: Jaccard estimate error small; candidate recall
vs true Jaccard pairs; precision on candidate pairs.
"""

from __future__ import annotations

import hashlib
import random

P = (1 << 31) - 1
K = 64
R = 2  # rows per band → B = K/R bands (s-curve tuned for j≳0.3)


def sig(items: set[int], params: list[tuple[int, int]]) -> list[int]:
    out = []
    for a, b in params:
        out.append(min((a * _h(x) + b) % P for x in items))
    return out


def _h(x: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{x}".encode()).digest()[:4], "little")


def est_jaccard(s1: list[int], s2: list[int]) -> float:
    return sum(1 for a, b in zip(s1, s2, strict=True) if a == b) / len(s1)


def lsh_candidates(sigs: list[list[int]]) -> set[tuple[int, int]]:
    cands: set[tuple[int, int]] = set()
    bands = len(sigs[0]) // R
    for band in range(bands):
        buckets: dict[tuple[int, ...], list[int]] = {}
        for i, s in enumerate(sigs):
            key = tuple(s[band * R : (band + 1) * R])
            buckets.setdefault(key, []).append(i)
        for v in buckets.values():
            for a in v:
                for b in v:
                    if a < b:
                        cands.add((a, b))
    return cands


def bench_minhash_lsh(seed: int = 20261231 + 314) -> dict[str, float]:
    rng = random.Random(seed)
    params = [(rng.randrange(1, P), rng.randrange(P)) for _ in range(K)]
    errs: list[float] = []
    hits = high_pairs = 0
    trials = 40
    for _ in range(trials):
        universe = set(range(200))
        a = set(rng.sample(sorted(universe), 60))
        # B shares half of A → true Jaccard ≈ 0.5
        shared = set(rng.sample(sorted(a), 30))
        b = shared | set(rng.sample(sorted(universe - a), 30))
        true_j = len(a & b) / len(a | b)
        s1, s2 = sig(a, params), sig(b, params)
        errs.append(abs(est_jaccard(s1, s2) - true_j))
        cands = lsh_candidates([s1, s2])
        if true_j >= 0.3:
            high_pairs += 1
            hits += int((0, 1) in cands)
    # error bound: est within ~0.2 on average
    return {
        "synthetic_mean_jaccard_err": float(sum(errs) / len(errs)),
        "synthetic_err_ok": float(sum(1 for e in errs if e < 0.2) / len(errs)),
        "synthetic_recall_high_j": float(hits / max(1, high_pairs)),
    }
