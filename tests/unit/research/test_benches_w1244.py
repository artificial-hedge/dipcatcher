"""Wave-1244 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1244 import (
    bench_ct_imaging_studies_family,
    bench_mammography_studies_family,
    bench_mri_studies_family,
    bench_neuroradiology_studies_family,
    bench_pet_imaging_studies_family,
    bench_ultrasound_studies_family,
)

_FAMILY_BENCHES = [
    bench_neuroradiology_studies_family,
    bench_mammography_studies_family,
    bench_ultrasound_studies_family,
    bench_ct_imaging_studies_family,
    bench_mri_studies_family,
    bench_pet_imaging_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
