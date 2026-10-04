"""Wave-918 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w918 import (
    bench_fractional_cascade_family,
    bench_free_list_family,
    bench_halfplane_isect_family,
    bench_object_pool_family,
    bench_range_min_query_family,
    bench_welzl_circle_family,
)

_FAMILY_BENCHES = [
    bench_fractional_cascade_family,
    bench_range_min_query_family,
    bench_free_list_family,
    bench_object_pool_family,
    bench_welzl_circle_family,
    bench_halfplane_isect_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
