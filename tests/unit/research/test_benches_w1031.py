"""Wave-1031 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1031 import (
    bench_construction_mgmt_family,
    bench_geotechnics_family,
    bench_structural_analysis_family,
    bench_surveying_family,
    bench_transportation_eng_family,
    bench_water_resources_family,
)

_FAMILY_BENCHES = [
    bench_structural_analysis_family,
    bench_geotechnics_family,
    bench_transportation_eng_family,
    bench_water_resources_family,
    bench_construction_mgmt_family,
    bench_surveying_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
