"""Wave-1256 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1256 import (
    bench_community_health_studies_family,
    bench_health_disparities_studies_family,
    bench_outbreak_studies_family,
    bench_screening_studies_family,
    bench_surveillance_studies_family,
    bench_vaccination_studies_family,
)

_FAMILY_BENCHES = [
    bench_screening_studies_family,
    bench_vaccination_studies_family,
    bench_outbreak_studies_family,
    bench_surveillance_studies_family,
    bench_health_disparities_studies_family,
    bench_community_health_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
