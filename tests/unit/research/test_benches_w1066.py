"""Wave-1066 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1066 import (
    bench_african_studies_family,
    bench_asian_studies_family,
    bench_european_studies_family,
    bench_latin_american_studies_family,
    bench_middle_eastern_studies_family,
    bench_slavic_studies_family,
)

_FAMILY_BENCHES = [
    bench_latin_american_studies_family,
    bench_asian_studies_family,
    bench_european_studies_family,
    bench_middle_eastern_studies_family,
    bench_african_studies_family,
    bench_slavic_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
