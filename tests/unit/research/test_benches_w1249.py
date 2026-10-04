"""Wave-1249 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1249 import (
    bench_acne_studies_family,
    bench_alopecia_studies_family,
    bench_eczema_studies_family,
    bench_psoriasis_studies_family,
    bench_skin_cancer_studies_family,
    bench_vitiligo_studies_family,
)

_FAMILY_BENCHES = [
    bench_skin_cancer_studies_family,
    bench_psoriasis_studies_family,
    bench_eczema_studies_family,
    bench_acne_studies_family,
    bench_vitiligo_studies_family,
    bench_alopecia_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
