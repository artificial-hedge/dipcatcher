"""Wave-928 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w928 import (
    bench_beta_skeleton_family,
    bench_convex_hull_3d_family,
    bench_medial_axis_family,
    bench_polygon_boolean_family,
    bench_polygon_centroid_family,
    bench_shape_context_family,
)

_FAMILY_BENCHES = [
    bench_convex_hull_3d_family,
    bench_polygon_boolean_family,
    bench_medial_axis_family,
    bench_polygon_centroid_family,
    bench_shape_context_family,
    bench_beta_skeleton_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
