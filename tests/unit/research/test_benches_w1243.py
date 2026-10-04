"""Wave-1243 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1243 import (
    bench_healthcare_infection_studies_family,
    bench_mycosis_studies_family,
    bench_opportunistic_studies_family,
    bench_sepsis_studies_family,
    bench_sexually_transmitted_studies_family,
    bench_tuberculosis_studies_family,
)

_FAMILY_BENCHES = [
    bench_sepsis_studies_family,
    bench_tuberculosis_studies_family,
    bench_mycosis_studies_family,
    bench_sexually_transmitted_studies_family,
    bench_healthcare_infection_studies_family,
    bench_opportunistic_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
