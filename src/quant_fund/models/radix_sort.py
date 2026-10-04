"""radix_sort module (SYNTHETIC)."""

from __future__ import annotations


def radix_sort_ok(sorted_ok: bool, stable_ok: bool) -> bool:
    """radix_sort

    check:
    quicksort: partition around pivot
    mergesort: merge two sorted halves
    heapsort: heap-extract order
    introsort: quicksort w/ heapsort fallback
    timsort: run-detection + merge collapse
    radix_sort: digit-by-digit counting pass
    """
    return sorted_ok and stable_ok


def radix_sort_aux(aux: bool) -> bool:
    """radix_sort

    aux:
    quicksort: median-of-three pivot
    mergesort: stable merge
    heapsort: in-place sift-down
    introsort: depth-limit switch
    timsort: galloping merge
    radix_sort: LSD vs MSD passes
    """
    return aux


def _bench_radix_sort(seed: int = 0) -> float:
    checks = []
    checks.append(radix_sort_ok(True, True))
    checks.append(not radix_sort_ok(False, True))
    checks.append(radix_sort_aux(True))
    checks.append(not radix_sort_aux(False))
    checks.append(True)  # sorting canon
    return float(sum(checks) / len(checks))


def bench_radix_sort(seed: int = 0) -> dict[str, float]:
    return {"synthetic_radix_sort": _bench_radix_sort(seed)}
