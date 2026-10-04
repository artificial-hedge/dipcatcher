"""Wave-1140 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1140 import (
    bench_glaciology_family,
    bench_hydrology_2_family,
    bench_oceanography_family,
    bench_paleoclimatology_family,
    bench_seismology_family,
    bench_volcanology_2_family,
)

_FAMILY_BENCHES = [
    bench_oceanography_family,
    bench_hydrology_2_family,
    bench_seismology_family,
    bench_glaciology_family,
    bench_paleoclimatology_family,
    bench_volcanology_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
