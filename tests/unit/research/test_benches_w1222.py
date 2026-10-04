"""Wave-1222 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1222 import (
    bench_hand_surgery_family,
    bench_joint_replacement_family,
    bench_musculoskeletal_medicine_family,
    bench_orthopedics_studies_family,
    bench_spine_surgery_family,
    bench_sports_medicine_orthopedics_family,
)

_FAMILY_BENCHES = [
    bench_orthopedics_studies_family,
    bench_sports_medicine_orthopedics_family,
    bench_musculoskeletal_medicine_family,
    bench_spine_surgery_family,
    bench_joint_replacement_family,
    bench_hand_surgery_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
