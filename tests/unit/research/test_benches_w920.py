"""Wave-920 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w920 import (
    bench_alpha_shape_family,
    bench_diameter_pair_family,
    bench_min_area_rect_family,
    bench_minkowski_sum_poly_family,
    bench_monotone_partition_family,
    bench_polygon_triangulate_family,
)

_FAMILY_BENCHES = [
    bench_monotone_partition_family,
    bench_polygon_triangulate_family,
    bench_min_area_rect_family,
    bench_diameter_pair_family,
    bench_alpha_shape_family,
    bench_minkowski_sum_poly_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
