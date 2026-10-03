"""Wave-919 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w919 import (
    bench_convex_layers_family,
    bench_delaunay_flip_family,
    bench_polygon_offset_family,
    bench_rotating_sweep_family,
    bench_visibility_graph_family,
    bench_voronoi_lite_family,
)

_FAMILY_BENCHES = [
    bench_voronoi_lite_family,
    bench_delaunay_flip_family,
    bench_convex_layers_family,
    bench_polygon_offset_family,
    bench_rotating_sweep_family,
    bench_visibility_graph_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
