"""Wave-1245 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1245 import (
    bench_breast_oncology_studies_family,
    bench_gi_oncology_studies_family,
    bench_immuno_oncology_studies_family,
    bench_medical_oncology_studies_family,
    bench_targeted_therapy_studies_family,
    bench_thoracic_oncology_studies_family,
)

_FAMILY_BENCHES = [
    bench_medical_oncology_studies_family,
    bench_immuno_oncology_studies_family,
    bench_targeted_therapy_studies_family,
    bench_breast_oncology_studies_family,
    bench_thoracic_oncology_studies_family,
    bench_gi_oncology_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
