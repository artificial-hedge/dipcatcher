"""Wave-1239 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1239 import (
    bench_celiac_studies_family,
    bench_gi_endoscopy_studies_family,
    bench_hepatology_medicine_family,
    bench_ibd_studies_family,
    bench_motility_studies_family,
    bench_pancreatic_medicine_family,
)

_FAMILY_BENCHES = [
    bench_gi_endoscopy_studies_family,
    bench_hepatology_medicine_family,
    bench_pancreatic_medicine_family,
    bench_ibd_studies_family,
    bench_celiac_studies_family,
    bench_motility_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
