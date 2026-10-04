"""Wave-1252 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1252 import (
    bench_hypothalamic_studies_family,
    bench_lipid_studies_family,
    bench_metabolic_syndrome_studies_family,
    bench_obesity_studies_family,
    bench_parathyroid_studies_family,
    bench_pituitary_studies_family,
)

_FAMILY_BENCHES = [
    bench_pituitary_studies_family,
    bench_parathyroid_studies_family,
    bench_lipid_studies_family,
    bench_obesity_studies_family,
    bench_metabolic_syndrome_studies_family,
    bench_hypothalamic_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
