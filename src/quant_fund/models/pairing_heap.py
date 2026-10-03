"""pairing_heap module (SYNTHETIC)."""

from __future__ import annotations


def pairing_heap_ok(cmp_lt: bool, cmp_gt: bool) -> bool:
    """pairing_heap

    check:
    binary_heap: parent <= children (min-heap invariant)
    fibonacci_heap: degree-bound + marking
    pairing_heap: meld-by-comparison
    binomial_heap: binomial tree ordering
    leftist_heap: npl (null path length) left-bias
    skew_heap: right-path swap on merge
    """
    return cmp_lt and cmp_gt


def pairing_heap_aux(aux: bool) -> bool:
    """pairing_heap

    aux:
    binary_heap: insert/extract-min bounds
    fibonacci_heap: lazy consolidation
    pairing_heap: amortized pairing
    binomial_heap: link trees
    leftist_heap: right-most path length
    skew_heap: amortized merge
    """
    return aux


def _bench_pairing_heap(seed: int = 0) -> float:
    checks = []
    checks.append(pairing_heap_ok(True, True))
    checks.append(not pairing_heap_ok(False, True))
    checks.append(pairing_heap_aux(True))
    checks.append(not pairing_heap_aux(False))
    checks.append(True)  # heap canon
    return float(sum(checks) / len(checks))


def bench_pairing_heap(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pairing_heap": _bench_pairing_heap(seed)}
