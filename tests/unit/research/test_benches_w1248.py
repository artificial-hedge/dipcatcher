"""Wave-1248 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1248 import (
    bench_cataract_studies_family,
    bench_corneal_studies_family,
    bench_glaucoma_studies_family,
    bench_macular_studies_family,
    bench_refractive_studies_family,
    bench_retinal_studies_family,
)

_FAMILY_BENCHES = [
    bench_retinal_studies_family,
    bench_corneal_studies_family,
    bench_glaucoma_studies_family,
    bench_cataract_studies_family,
    bench_macular_studies_family,
    bench_refractive_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
