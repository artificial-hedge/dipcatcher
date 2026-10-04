"""Wave-1065 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1065 import (
    bench_cartography_family,
    bench_climatology_family,
    bench_geomorphology_family,
    bench_human_geography_family,
    bench_physical_geography_family,
    bench_remote_sensing_family,
)

_FAMILY_BENCHES = [
    bench_physical_geography_family,
    bench_human_geography_family,
    bench_cartography_family,
    bench_remote_sensing_family,
    bench_geomorphology_family,
    bench_climatology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
