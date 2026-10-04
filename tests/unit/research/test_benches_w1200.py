"""Wave-1200 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1200 import (
    bench_cardiovascular_technology_family,
    bench_dosimetry_studies_family,
    bench_medical_physics_studies_family,
    bench_nuclear_medicine_technology_family,
    bench_radiation_dosimetry_family,
    bench_radiopharmacy_family,
)

_FAMILY_BENCHES = [
    bench_cardiovascular_technology_family,
    bench_nuclear_medicine_technology_family,
    bench_radiation_dosimetry_family,
    bench_medical_physics_studies_family,
    bench_dosimetry_studies_family,
    bench_radiopharmacy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
