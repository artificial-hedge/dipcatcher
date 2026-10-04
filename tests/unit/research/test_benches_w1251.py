"""Wave-1251 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1251 import (
    bench_andrology_studies_family,
    bench_bladder_studies_family,
    bench_bph_studies_family,
    bench_erectile_studies_family,
    bench_incontinence_studies_family,
    bench_prostate_studies_family,
)

_FAMILY_BENCHES = [
    bench_prostate_studies_family,
    bench_bladder_studies_family,
    bench_andrology_studies_family,
    bench_erectile_studies_family,
    bench_incontinence_studies_family,
    bench_bph_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
