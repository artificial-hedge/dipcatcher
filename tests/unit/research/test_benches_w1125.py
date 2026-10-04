"""Wave-1125 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1125 import (
    bench_agricultural_economics_family,
    bench_development_economics_family,
    bench_energy_economics_family,
    bench_environmental_economics_family,
    bench_health_economics_family,
    bench_urban_economics_family,
)

_FAMILY_BENCHES = [
    bench_development_economics_family,
    bench_environmental_economics_family,
    bench_health_economics_family,
    bench_urban_economics_family,
    bench_agricultural_economics_family,
    bench_energy_economics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
