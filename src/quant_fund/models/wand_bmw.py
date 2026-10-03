"""SYNTHETIC WAND (weak AND) top-k retrieval.

Classic Broder-style WAND: terms sorted by current docid, pivot found
by accumulating upper bounds past the threshold; only fully-verified
documents are scored. Top-k set verified against exhaustive scoring.
"""

from __future__ import annotations

import random

EVALED = 0


def wand_topk(
    postings: dict[str, list[int]],
    tfs: dict[str, dict[int, int]],
    terms: list[str],
    k: int,
) -> list[int]:
    global EVALED
    EVALED = 0
    ub = {t: max(tfs[t].values(), default=0) for t in terms}
    ptr = {t: 0 for t in terms}
    heap: list[tuple[int, int]] = []

    def cur(t: str) -> int | None:
        p = ptr[t]
        return postings[t][p] if p < len(postings[t]) else None

    def advance(t: str, docid: int) -> None:
        while ptr[t] < len(postings[t]) and postings[t][ptr[t]] < docid:
            ptr[t] += 1

    while True:
        live = [t for t in terms if cur(t) is not None]
        if not live:
            break
        live.sort(key=lambda t: cur(t) or 0)
        thresh = heap[0][0] if len(heap) == k else -1
        acc, pivot = 0.0, None
        for t in live:
            acc += ub[t]
            if acc >= thresh if len(heap) == k else acc > thresh:
                pivot = cur(t)
                break
        if pivot is None:
            break
        if cur(live[0]) == pivot:
            # all live terms' pointers ≤ pivot; those equal to pivot match
            EVALED += 1
            s = sum(tfs[t].get(pivot, 0) for t in terms)
            if len(heap) < k:
                heap.append((s, pivot))
                heap.sort(key=lambda x: (x[0], -x[1]))
            elif (s, -pivot) > (heap[0][0], -heap[0][1]):
                heap[0] = (s, pivot)
                heap.sort(key=lambda x: (x[0], -x[1]))
            for t in terms:
                if cur(t) == pivot:
                    ptr[t] += 1
        else:
            # advance terms whose pointer is before pivot
            for t in live:
                cur_t = cur(t)
                if cur_t is not None and cur_t < pivot:
                    advance(t, pivot)
    return [d for _, d in sorted(heap, key=lambda x: (-x[0], x[1]))]


def _exhaustive(tfs: dict[str, dict[int, int]], terms: list[str], k: int) -> list[int]:
    cand = set()
    for t in terms:
        cand |= set(tfs[t])
    scored = sorted(cand, key=lambda d: (-sum(tfs[t].get(d, 0) for t in terms), d))
    return scored[:k]


def bench_wand_bmw(seed: int = 20261231 + 462) -> dict[str, float]:
    rng = random.Random(seed)
    match = pruned = 0
    trials = 40
    for _ in range(trials):
        terms = rng.sample([f"t{i}" for i in range(12)], 4)
        nd = 60
        tfs = {t: {d: rng.randrange(0, 8) for d in range(nd) if rng.random() < 0.3} for t in terms}
        postings = {t: sorted(tfs[t]) for t in terms}
        k = rng.randrange(1, 6)
        match += int(wand_topk(postings, tfs, terms, k) == _exhaustive(tfs, terms, k))
        total_docs = len(set().union(*(set(tfs[t]) for t in terms)))
        pruned += int(total_docs >= EVALED)
    return {
        "synthetic_wand_topk_exact": float(match / trials),
        "synthetic_evals_bounded": float(pruned / trials),
    }
