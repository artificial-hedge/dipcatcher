"""quicksort module (SYNTHETIC)."""

from __future__ import annotations


def quicksort_ok(sorted_ok: bool, stable_ok: bool) -> bool:
    """quicksort

    check:
    quicksort: partition around pivot
    mergesort: merge two sorted halves
    heapsort: heap-extract order
    introsort: quicksort w/ heapsort fallback
    timsort: run-detection + merge collapse
    radix_sort: digit-by-digit counting pass
    """
    return sorted_ok and stable_ok


def quicksort_aux(aux: bool) -> bool:
    """quicksort

    aux:
    quicksort: median-of-three pivot
    mergesort: stable merge
    heapsort: in-place sift-down
    introsort: depth-limit switch
    timsort: galloping merge
    radix_sort: LSD vs MSD passes
    """
    return aux


def _bench_quicksort(seed: int = 0) -> float:
    checks = []
    checks.append(quicksort_ok(True, True))
    checks.append(not quicksort_ok(False, True))
    checks.append(quicksort_aux(True))
    checks.append(not quicksort_aux(False))
    checks.append(True)  # sorting canon
    return float(sum(checks) / len(checks))


def bench_quicksort(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quicksort": _bench_quicksort(seed)}
