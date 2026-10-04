"""Wave-1211 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1211 import (
    bench_epilepsy_studies_family,
    bench_headache_medicine_family,
    bench_movement_disorders_family,
    bench_neurodevelopmental_disorders_family,
    bench_neuropsychiatry_studies_family,
    bench_pediatric_neurology_family,
)

_FAMILY_BENCHES = [
    bench_pediatric_neurology_family,
    bench_neurodevelopmental_disorders_family,
    bench_neuropsychiatry_studies_family,
    bench_headache_medicine_family,
    bench_epilepsy_studies_family,
    bench_movement_disorders_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
