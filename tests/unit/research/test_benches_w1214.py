"""Wave-1214 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1214 import (
    bench_cardiology_studies_family,
    bench_cardiovascular_imaging_family,
    bench_electrophysiology_studies_family,
    bench_heart_failure_medicine_family,
    bench_interventional_cardiology_family,
    bench_preventive_cardiology_family,
)

_FAMILY_BENCHES = [
    bench_cardiology_studies_family,
    bench_interventional_cardiology_family,
    bench_electrophysiology_studies_family,
    bench_heart_failure_medicine_family,
    bench_preventive_cardiology_family,
    bench_cardiovascular_imaging_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
