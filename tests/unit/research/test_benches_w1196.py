"""Wave-1196 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1196 import (
    bench_disaster_management_family,
    bench_emergency_medical_technician_family,
    bench_fire_science_studies_family,
    bench_industrial_hygiene_family,
    bench_occupational_safety_family,
    bench_paramedic_studies_family,
)

_FAMILY_BENCHES = [
    bench_emergency_medical_technician_family,
    bench_fire_science_studies_family,
    bench_paramedic_studies_family,
    bench_disaster_management_family,
    bench_occupational_safety_family,
    bench_industrial_hygiene_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
