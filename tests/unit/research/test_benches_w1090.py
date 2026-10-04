"""Wave-1090 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1090 import (
    bench_history_of_science_family,
    bench_information_history_family,
    bench_media_archaeology_family,
    bench_philosophy_of_technology_family,
    bench_sts_studies_family,
    bench_technology_studies_family,
)

_FAMILY_BENCHES = [
    bench_history_of_science_family,
    bench_sts_studies_family,
    bench_philosophy_of_technology_family,
    bench_media_archaeology_family,
    bench_information_history_family,
    bench_technology_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
