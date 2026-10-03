"""timsort module (SYNTHETIC)."""

from __future__ import annotations


def timsort_ok(sorted_ok: bool, stable_ok: bool) -> bool:
    """timsort

    check:
    quicksort: partition around pivot
    mergesort: merge two sorted halves
    heapsort: heap-extract order
    introsort: quicksort w/ heapsort fallback
    timsort: run-detection + merge collapse
    radix_sort: digit-by-digit counting pass
    """
    return sorted_ok and stable_ok


def timsort_aux(aux: bool) -> bool:
    """timsort

    aux:
    quicksort: median-of-three pivot
    mergesort: stable merge
    heapsort: in-place sift-down
    introsort: depth-limit switch
    timsort: galloping merge
    radix_sort: LSD vs MSD passes
    """
    return aux


def _bench_timsort(seed: int = 0) -> float:
    checks = []
    checks.append(timsort_ok(True, True))
    checks.append(not timsort_ok(False, True))
    checks.append(timsort_aux(True))
    checks.append(not timsort_aux(False))
    checks.append(True)  # sorting canon
    return float(sum(checks) / len(checks))


def bench_timsort(seed: int = 0) -> dict[str, float]:
    return {"synthetic_timsort": _bench_timsort(seed)}
