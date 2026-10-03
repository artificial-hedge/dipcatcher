"""SYNTHETIC MinHash-LSH near-duplicate detection.

Signature bands → candidate pairs, then exact Jaccard confirm.
Recall = fraction of true duplicate pairs (J≥0.6) recovered.
"""

from __future__ import annotations

import random


def _minhash(shingles: set[int], n_perm: int, seed: int) -> list[int]:
    out = []
    for p in range(n_perm):
        a = (seed * 2654435761 + p * 40503) & 0xFFFFFFFF
        b = (seed + p * 99991) & 0xFFFFFFFF
        out.append(min(((a * s + b) & 0xFFFFFFFF) for s in shingles))
    return out


def _lsh_pairs(sigs: list[list[int]], bands: int) -> set[tuple[int, int]]:
    rows = len(sigs[0]) // bands
    cands: set[tuple[int, int]] = set()
    for b in range(bands):
        buckets: dict[tuple, list[int]] = {}
        for i, sig in enumerate(sigs):
            key = tuple(sig[b * rows : (b + 1) * rows])
            buckets.setdefault(key, []).append(i)
        for ids in buckets.values():
            if len(ids) > 1:
                for x in range(len(ids)):
                    for y in range(x + 1, len(ids)):
                        cands.add((ids[x], ids[y]))
    return cands


def _jac(a: set[int], b: set[int]) -> float:
    return len(a & b) / max(1, len(a | b))


def bench_lsh_dedup(seed: int = 20261231 + 463) -> dict[str, float]:
    rng = random.Random(seed)
    rec_total = prec_total = 0.0
    trials = 30
    for _ in range(trials):
        base = set(rng.sample(range(2000), 40))
        docs = [set(rng.sample(range(2000), rng.randrange(10, 50))) for _ in range(20)]
        # inject near-dups
        for i in range(0, 20, 5):
            drop = set(rng.sample(list(base), 8))
            docs[i] = (base - drop) | set(rng.sample(range(2000), 6))
        truth = {
            (i, j) for i in range(20) for j in range(i + 1, 20) if _jac(docs[i], docs[j]) >= 0.5
        }
        sigs = [_minhash(d, 60, seed) for d in docs]
        cands = _lsh_pairs(sigs, 12)
        found = {p for p in cands if _jac(docs[p[0]], docs[p[1]]) >= 0.5}
        if truth:
            rec_total += len(found & truth) / len(truth)
            prec_total += len(found & truth) / max(1, len(cands))
    n = sum(1 for _ in range(trials))
    return {
        "synthetic_duplicate_recall": float(rec_total / n),
        "synthetic_candidate_precision": float(prec_total / n),
    }
