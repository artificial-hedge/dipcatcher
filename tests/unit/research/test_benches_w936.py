"""Wave-936 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w936 import (
    bench_bundle_level_family,
    bench_clarke_subdiff_family,
    bench_epigraph_proj_family,
    bench_gauge_duality_family,
    bench_gauge_fn_family,
    bench_subdiff_compute_family,
)

_FAMILY_BENCHES = [
    bench_subdiff_compute_family,
    bench_epigraph_proj_family,
    bench_gauge_fn_family,
    bench_gauge_duality_family,
    bench_bundle_level_family,
    bench_clarke_subdiff_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
