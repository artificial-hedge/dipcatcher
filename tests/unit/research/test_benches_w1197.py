"""Wave-1197 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1197 import (
    bench_addiction_counseling_family,
    bench_genetic_screening_family,
    bench_neonatology_studies_family,
    bench_pediatric_therapeutics_family,
    bench_prenatal_studies_family,
    bench_rehabilitation_counseling_family,
)

_FAMILY_BENCHES = [
    bench_addiction_counseling_family,
    bench_rehabilitation_counseling_family,
    bench_genetic_screening_family,
    bench_prenatal_studies_family,
    bench_neonatology_studies_family,
    bench_pediatric_therapeutics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
