"""introsort module (SYNTHETIC)."""

from __future__ import annotations


def introsort_ok(sorted_ok: bool, stable_ok: bool) -> bool:
    """introsort

    check:
    quicksort: partition around pivot
    mergesort: merge two sorted halves
    heapsort: heap-extract order
    introsort: quicksort w/ heapsort fallback
    timsort: run-detection + merge collapse
    radix_sort: digit-by-digit counting pass
    """
    return sorted_ok and stable_ok


def introsort_aux(aux: bool) -> bool:
    """introsort

    aux:
    quicksort: median-of-three pivot
    mergesort: stable merge
    heapsort: in-place sift-down
    introsort: depth-limit switch
    timsort: galloping merge
    radix_sort: LSD vs MSD passes
    """
    return aux


def _bench_introsort(seed: int = 0) -> float:
    checks = []
    checks.append(introsort_ok(True, True))
    checks.append(not introsort_ok(False, True))
    checks.append(introsort_aux(True))
    checks.append(not introsort_aux(False))
    checks.append(True)  # sorting canon
    return float(sum(checks) / len(checks))


def bench_introsort(seed: int = 0) -> dict[str, float]:
    return {"synthetic_introsort": _bench_introsort(seed)}
