"""Wave-1189 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1189 import (
    bench_event_management_family,
    bench_hospitality_studies_family,
    bench_hotel_management_family,
    bench_leisure_science_family,
    bench_recreation_management_family,
    bench_tourism_studies_family,
)

_FAMILY_BENCHES = [
    bench_hospitality_studies_family,
    bench_event_management_family,
    bench_hotel_management_family,
    bench_tourism_studies_family,
    bench_recreation_management_family,
    bench_leisure_science_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
