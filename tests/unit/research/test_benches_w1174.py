"""Wave-1174 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1174 import (
    bench_hospitality_family,
    bench_leisure_studies_family,
    bench_recreation_family,
    bench_recreation_therapy_family,
    bench_sports_management_family,
    bench_tourism_family,
)

_FAMILY_BENCHES = [
    bench_recreation_family,
    bench_leisure_studies_family,
    bench_tourism_family,
    bench_hospitality_family,
    bench_sports_management_family,
    bench_recreation_therapy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
