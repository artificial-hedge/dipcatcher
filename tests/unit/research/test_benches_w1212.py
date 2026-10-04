"""Wave-1212 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1212 import (
    bench_neuro_ophthalmology_family,
    bench_neurocritical_care_family,
    bench_neurogenetics_family,
    bench_neuroimmunology_family,
    bench_neuromuscular_medicine_family,
    bench_neurovascular_studies_family,
)

_FAMILY_BENCHES = [
    bench_neurocritical_care_family,
    bench_neurovascular_studies_family,
    bench_neuromuscular_medicine_family,
    bench_neuro_ophthalmology_family,
    bench_neuroimmunology_family,
    bench_neurogenetics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
