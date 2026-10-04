"""Wave-1232 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1232 import (
    bench_caregiver_medicine_family,
    bench_falls_prevention_studies_family,
    bench_frailty_medicine_family,
    bench_geriatrics_studies_family,
    bench_memory_clinic_studies_family,
    bench_polypharmacy_studies_family,
)

_FAMILY_BENCHES = [
    bench_geriatrics_studies_family,
    bench_frailty_medicine_family,
    bench_memory_clinic_studies_family,
    bench_falls_prevention_studies_family,
    bench_polypharmacy_studies_family,
    bench_caregiver_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
