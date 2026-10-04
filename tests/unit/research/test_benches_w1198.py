"""Wave-1198 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1198 import (
    bench_electrodiagnostic_studies_family,
    bench_hyperbaric_medicine_family,
    bench_infusion_therapy_family,
    bench_pain_management_family,
    bench_sleep_medicine_family,
    bench_wound_care_family,
)

_FAMILY_BENCHES = [
    bench_sleep_medicine_family,
    bench_pain_management_family,
    bench_wound_care_family,
    bench_infusion_therapy_family,
    bench_hyperbaric_medicine_family,
    bench_electrodiagnostic_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
