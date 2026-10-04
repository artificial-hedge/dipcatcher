"""Wave-1246 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1246 import (
    bench_bariatric_surgery_studies_family,
    bench_burn_surgery_studies_family,
    bench_endocrine_surgery_studies_family,
    bench_pediatric_surgery_studies_family,
    bench_plastic_surgery_studies_family,
    bench_transplant_surgery_studies_family,
)

_FAMILY_BENCHES = [
    bench_bariatric_surgery_studies_family,
    bench_pediatric_surgery_studies_family,
    bench_plastic_surgery_studies_family,
    bench_burn_surgery_studies_family,
    bench_endocrine_surgery_studies_family,
    bench_transplant_surgery_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
