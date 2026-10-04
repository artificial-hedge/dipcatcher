"""Wave-1223 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1223 import (
    bench_airway_management_family,
    bench_anesthesiology_studies_family,
    bench_pain_medicine_studies_family,
    bench_perioperative_medicine_family,
    bench_regional_anesthesia_family,
    bench_sedation_medicine_family,
)

_FAMILY_BENCHES = [
    bench_anesthesiology_studies_family,
    bench_perioperative_medicine_family,
    bench_pain_medicine_studies_family,
    bench_regional_anesthesia_family,
    bench_sedation_medicine_family,
    bench_airway_management_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
