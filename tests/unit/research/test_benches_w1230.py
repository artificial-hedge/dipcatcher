"""Wave-1230 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1230 import (
    bench_acute_care_studies_family,
    bench_disaster_medicine_family,
    bench_emergency_medicine_studies_family,
    bench_resuscitation_medicine_family,
    bench_toxicology_medicine_family,
    bench_trauma_medicine_family,
)

_FAMILY_BENCHES = [
    bench_emergency_medicine_studies_family,
    bench_trauma_medicine_family,
    bench_toxicology_medicine_family,
    bench_disaster_medicine_family,
    bench_acute_care_studies_family,
    bench_resuscitation_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
