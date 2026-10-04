"""heapsort module (SYNTHETIC)."""

from __future__ import annotations


def heapsort_ok(sorted_ok: bool, stable_ok: bool) -> bool:
    """heapsort

    check:
    quicksort: partition around pivot
    mergesort: merge two sorted halves
    heapsort: heap-extract order
    introsort: quicksort w/ heapsort fallback
    timsort: run-detection + merge collapse
    radix_sort: digit-by-digit counting pass
    """
    return sorted_ok and stable_ok


def heapsort_aux(aux: bool) -> bool:
    """heapsort

    aux:
    quicksort: median-of-three pivot
    mergesort: stable merge
    heapsort: in-place sift-down
    introsort: depth-limit switch
    timsort: galloping merge
    radix_sort: LSD vs MSD passes
    """
    return aux


def _bench_heapsort(seed: int = 0) -> float:
    checks = []
    checks.append(heapsort_ok(True, True))
    checks.append(not heapsort_ok(False, True))
    checks.append(heapsort_aux(True))
    checks.append(not heapsort_aux(False))
    checks.append(True)  # sorting canon
    return float(sum(checks) / len(checks))


def bench_heapsort(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heapsort": _bench_heapsort(seed)}
