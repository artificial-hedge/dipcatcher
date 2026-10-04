"""Wave-1242 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1242 import (
    bench_fracture_studies_family,
    bench_osteoporosis_studies_family,
    bench_physiatry_studies_family,
    bench_physical_therapy_studies_family,
    bench_rehabilitation_studies_family,
    bench_sports_injury_studies_family,
)

_FAMILY_BENCHES = [
    bench_rehabilitation_studies_family,
    bench_physical_therapy_studies_family,
    bench_sports_injury_studies_family,
    bench_fracture_studies_family,
    bench_osteoporosis_studies_family,
    bench_physiatry_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
