"""Wave-1173 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1173 import (
    bench_communication_3_family,
    bench_digital_media_2_family,
    bench_information_science_3_family,
    bench_journalism_3_family,
    bench_media_studies_3_family,
    bench_rhetoric_2_family,
)

_FAMILY_BENCHES = [
    bench_communication_3_family,
    bench_journalism_3_family,
    bench_media_studies_3_family,
    bench_rhetoric_2_family,
    bench_information_science_3_family,
    bench_digital_media_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
