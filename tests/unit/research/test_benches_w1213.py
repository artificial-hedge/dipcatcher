"""Wave-1213 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1213 import (
    bench_neurorehabilitation_family,
    bench_neurosurgery_studies_family,
    bench_neurotoxicology_family,
    bench_neurotrauma_family,
    bench_neurovascular_surgery_family,
    bench_spinal_cord_medicine_family,
)

_FAMILY_BENCHES = [
    bench_neurosurgery_studies_family,
    bench_neurotrauma_family,
    bench_neurotoxicology_family,
    bench_neurorehabilitation_family,
    bench_neurovascular_surgery_family,
    bench_spinal_cord_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
