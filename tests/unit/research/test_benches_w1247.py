"""Wave-1247 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1247 import (
    bench_aki_studies_family,
    bench_ckd_studies_family,
    bench_electrolyte_studies_family,
    bench_glomerular_studies_family,
    bench_stones_studies_family,
    bench_tubulointerstitial_studies_family,
)

_FAMILY_BENCHES = [
    bench_glomerular_studies_family,
    bench_tubulointerstitial_studies_family,
    bench_ckd_studies_family,
    bench_aki_studies_family,
    bench_electrolyte_studies_family,
    bench_stones_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
