"""Wave-1216 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1216 import (
    bench_critical_care_medicine_family,
    bench_gastroenterology_studies_family,
    bench_hepatology_studies_family,
    bench_hospital_medicine_family,
    bench_internal_medicine_family,
    bench_pulmonary_medicine_family,
)

_FAMILY_BENCHES = [
    bench_internal_medicine_family,
    bench_hospital_medicine_family,
    bench_critical_care_medicine_family,
    bench_pulmonary_medicine_family,
    bench_gastroenterology_studies_family,
    bench_hepatology_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
