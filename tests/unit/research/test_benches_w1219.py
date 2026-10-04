"""Wave-1219 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1219 import (
    bench_hematologic_malignancies_family,
    bench_hematology_studies_family,
    bench_oncology_studies_family,
    bench_radiation_oncology_family,
    bench_solid_tumor_oncology_family,
    bench_transfusion_medicine_family,
)

_FAMILY_BENCHES = [
    bench_hematology_studies_family,
    bench_oncology_studies_family,
    bench_hematologic_malignancies_family,
    bench_solid_tumor_oncology_family,
    bench_transfusion_medicine_family,
    bench_radiation_oncology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
