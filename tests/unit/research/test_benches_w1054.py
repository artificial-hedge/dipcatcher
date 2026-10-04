"""Wave-1054 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1054 import (
    bench_criminology_family,
    bench_demography_family,
    bench_economic_sociology_family,
    bench_social_networks_family,
    bench_social_stratification_family,
    bench_urban_sociology_family,
)

_FAMILY_BENCHES = [
    bench_social_networks_family,
    bench_demography_family,
    bench_criminology_family,
    bench_urban_sociology_family,
    bench_economic_sociology_family,
    bench_social_stratification_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
