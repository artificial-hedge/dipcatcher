"""Wave-1070 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1070 import (
    bench_athletic_training_family,
    bench_exercise_physiology_family,
    bench_sports_analytics_family,
    bench_sports_biomechanics_family,
    bench_sports_psychology_family,
    bench_sports_science_family,
)

_FAMILY_BENCHES = [
    bench_sports_science_family,
    bench_exercise_physiology_family,
    bench_sports_biomechanics_family,
    bench_sports_psychology_family,
    bench_athletic_training_family,
    bench_sports_analytics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
