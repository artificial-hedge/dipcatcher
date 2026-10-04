"""Wave-1237 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1237 import (
    bench_clinical_pharmacy_studies_family,
    bench_compounding_pharmacy_family,
    bench_hospital_pharmacy_studies_family,
    bench_medication_therapy_mgmt_family,
    bench_pharmacovigilance_studies_family,
    bench_pharmacy_practice_studies_family,
)

_FAMILY_BENCHES = [
    bench_clinical_pharmacy_studies_family,
    bench_pharmacy_practice_studies_family,
    bench_medication_therapy_mgmt_family,
    bench_compounding_pharmacy_family,
    bench_pharmacovigilance_studies_family,
    bench_hospital_pharmacy_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
