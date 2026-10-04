"""Wave-1231 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1231 import (
    bench_allergy_studies_family,
    bench_autoimmunity_studies_family,
    bench_hematopoietic_transplant_family,
    bench_immunodeficiency_studies_family,
    bench_immunology_medicine_family,
    bench_transplant_medicine_studies_family,
)

_FAMILY_BENCHES = [
    bench_transplant_medicine_studies_family,
    bench_immunology_medicine_family,
    bench_allergy_studies_family,
    bench_autoimmunity_studies_family,
    bench_hematopoietic_transplant_family,
    bench_immunodeficiency_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
