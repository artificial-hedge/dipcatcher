"""Marking algorithm for k-paging (SYNTHETIC).

Phases: unmark all at phase start; on fault, evict an unmarked page,
mark the requested one. Deterministic version evicts the lowest-index
unmarked page. Bench compares faults against LRU and Belady's optimal
on a fixed reference string with k=3.
"""


def _refs() -> list[int]:
    return [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 6, 2, 4]


def _marking(refs: list[int], k: int) -> int:
    cache: list[int] = []
    marked: set[int] = set()
    faults = 0
    for p in refs:
        if p in cache:
            marked.add(p)
            continue
        faults += 1
        if len(cache) == k:
            unmarked = [q for q in cache if q not in marked]
            if not unmarked:
                marked = set()
                unmarked = list(cache)
            cache.remove(unmarked[0])
        cache.append(p)
        marked.add(p)
    return faults


def _lru(refs: list[int], k: int) -> int:
    cache: list[int] = []
    faults = 0
    for p in refs:
        if p in cache:
            cache.remove(p)
            cache.append(p)
            continue
        faults += 1
        if len(cache) == k:
            cache.pop(0)
        cache.append(p)
    return faults


def _belady(refs: list[int], k: int) -> int:
    cache: list[int] = []
    faults = 0
    for i, p in enumerate(refs):
        if p in cache:
            continue
        faults += 1
        if len(cache) == k:
            nxt = {q: (refs[i + 1 :].index(q) if q in refs[i + 1 :] else 10**9) for q in cache}
            cache.remove(max(nxt, key=lambda q: nxt[q]))
        cache.append(p)
    return faults


def bench_marking_paging(seed: int = 5403) -> dict[str, float]:
    refs, k = _refs(), 3
    m, lru_f, b = _marking(refs, k), _lru(refs, k), _belady(refs, k)
    return {
        "synthetic_page_marking": float(m),
        "synthetic_page_lru": float(lru_f),
        "synthetic_page_belady": float(b),
        "synthetic_page_marking_ratio": m / b,
        "synthetic_page_k_bound": float(k),
    }
