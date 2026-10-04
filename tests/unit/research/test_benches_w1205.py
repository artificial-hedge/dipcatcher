"""Wave-1205 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1205 import (
    bench_aerospace_medicine_family,
    bench_diving_medicine_family,
    bench_high_altitude_medicine_family,
    bench_hyperbaric_oxygen_family,
    bench_space_physiology_family,
    bench_wilderness_medicine_family,
)

_FAMILY_BENCHES = [
    bench_aerospace_medicine_family,
    bench_diving_medicine_family,
    bench_wilderness_medicine_family,
    bench_space_physiology_family,
    bench_hyperbaric_oxygen_family,
    bench_high_altitude_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
