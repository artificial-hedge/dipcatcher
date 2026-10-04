"""Wave-1113 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1113 import (
    bench_anesthesiology_family,
    bench_emergency_medicine_family,
    bench_family_medicine_family,
    bench_obstetrics_gynecology_family,
    bench_pediatrics_family,
    bench_surgery_family,
)

_FAMILY_BENCHES = [
    bench_surgery_family,
    bench_anesthesiology_family,
    bench_obstetrics_gynecology_family,
    bench_pediatrics_family,
    bench_emergency_medicine_family,
    bench_family_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
