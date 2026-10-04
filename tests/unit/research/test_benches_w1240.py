"""Wave-1240 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1240 import (
    bench_asthma_studies_family,
    bench_bronchiectasis_studies_family,
    bench_copd_studies_family,
    bench_interstitial_lung_studies_family,
    bench_respiratory_studies_family,
    bench_sleep_breathing_studies_family,
)

_FAMILY_BENCHES = [
    bench_respiratory_studies_family,
    bench_asthma_studies_family,
    bench_copd_studies_family,
    bench_interstitial_lung_studies_family,
    bench_sleep_breathing_studies_family,
    bench_bronchiectasis_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
