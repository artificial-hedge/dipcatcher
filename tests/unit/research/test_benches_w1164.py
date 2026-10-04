"""Wave-1164 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1164 import (
    bench_communication_studies_2_family,
    bench_education_5_family,
    bench_information_science_2_family,
    bench_journalism_2_family,
    bench_library_science_2_family,
    bench_media_studies_2_family,
)

_FAMILY_BENCHES = [
    bench_education_5_family,
    bench_communication_studies_2_family,
    bench_media_studies_2_family,
    bench_journalism_2_family,
    bench_library_science_2_family,
    bench_information_science_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
