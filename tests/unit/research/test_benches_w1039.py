"""Wave-1039 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1039 import (
    bench_air_pollution_control_family,
    bench_environmental_remediation_family,
    bench_noise_control_family,
    bench_waste_management_family,
    bench_wastewater_engineering_family,
    bench_water_treatment_family,
)

_FAMILY_BENCHES = [
    bench_water_treatment_family,
    bench_air_pollution_control_family,
    bench_waste_management_family,
    bench_environmental_remediation_family,
    bench_wastewater_engineering_family,
    bench_noise_control_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
