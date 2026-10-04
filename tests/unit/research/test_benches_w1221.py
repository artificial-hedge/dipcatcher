"""Wave-1221 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1221 import (
    bench_audiology_medicine_family,
    bench_dermatology_studies_family,
    bench_dermatopathology_family,
    bench_ophthalmology_studies_family,
    bench_optometry_studies_family,
    bench_otolaryngology_studies_family,
)

_FAMILY_BENCHES = [
    bench_dermatology_studies_family,
    bench_ophthalmology_studies_family,
    bench_otolaryngology_studies_family,
    bench_audiology_medicine_family,
    bench_optometry_studies_family,
    bench_dermatopathology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
