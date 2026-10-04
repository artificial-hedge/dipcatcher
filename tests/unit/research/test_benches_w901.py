"""Wave-901 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w901 import (
    bench_binary_heap_family,
    bench_binomial_heap_family,
    bench_fibonacci_heap_family,
    bench_leftist_heap_family,
    bench_pairing_heap_family,
    bench_skew_heap_family,
)

_FAMILY_BENCHES = [
    bench_binary_heap_family,
    bench_fibonacci_heap_family,
    bench_pairing_heap_family,
    bench_binomial_heap_family,
    bench_leftist_heap_family,
    bench_skew_heap_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
