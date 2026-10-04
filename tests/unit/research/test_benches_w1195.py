"""Wave-1195 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1195 import (
    bench_clinical_laboratory_family,
    bench_medical_imaging_studies_family,
    bench_mortuary_science_family,
    bench_phlebotomy_studies_family,
    bench_sterile_processing_family,
    bench_surgical_technology_family,
)

_FAMILY_BENCHES = [
    bench_medical_imaging_studies_family,
    bench_clinical_laboratory_family,
    bench_mortuary_science_family,
    bench_phlebotomy_studies_family,
    bench_surgical_technology_family,
    bench_sterile_processing_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
