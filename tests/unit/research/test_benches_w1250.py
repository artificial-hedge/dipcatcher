"""Wave-1250 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1250 import (
    bench_cochlear_studies_family,
    bench_head_neck_surgery_studies_family,
    bench_laryngology_studies_family,
    bench_otology_studies_family,
    bench_rhinology_studies_family,
    bench_sinus_studies_family,
)

_FAMILY_BENCHES = [
    bench_sinus_studies_family,
    bench_laryngology_studies_family,
    bench_otology_studies_family,
    bench_rhinology_studies_family,
    bench_head_neck_surgery_studies_family,
    bench_cochlear_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
