"""Wave-905 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w905 import (
    bench_heapsort_family,
    bench_introsort_family,
    bench_mergesort_family,
    bench_quicksort_family,
    bench_radix_sort_family,
    bench_timsort_family,
)

_FAMILY_BENCHES = [
    bench_quicksort_family,
    bench_mergesort_family,
    bench_heapsort_family,
    bench_introsort_family,
    bench_timsort_family,
    bench_radix_sort_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
