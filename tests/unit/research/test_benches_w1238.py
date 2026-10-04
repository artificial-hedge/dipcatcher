"""Wave-1238 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1238 import (
    bench_aortic_medicine_studies_family,
    bench_lymphatic_medicine_family,
    bench_peripheral_artery_studies_family,
    bench_phlebology_studies_family,
    bench_vascular_lab_studies_family,
    bench_vascular_medicine_studies_family,
)

_FAMILY_BENCHES = [
    bench_vascular_medicine_studies_family,
    bench_phlebology_studies_family,
    bench_lymphatic_medicine_family,
    bench_vascular_lab_studies_family,
    bench_peripheral_artery_studies_family,
    bench_aortic_medicine_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
