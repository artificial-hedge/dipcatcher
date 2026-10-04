"""Wave-1220 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1220 import (
    bench_allergy_immunology_family,
    bench_antimicrobial_stewardship_family,
    bench_hiv_medicine_family,
    bench_immunology_studies_family,
    bench_infectious_disease_medicine_family,
    bench_rheumatology_studies_family,
)

_FAMILY_BENCHES = [
    bench_infectious_disease_medicine_family,
    bench_hiv_medicine_family,
    bench_antimicrobial_stewardship_family,
    bench_rheumatology_studies_family,
    bench_immunology_studies_family,
    bench_allergy_immunology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
