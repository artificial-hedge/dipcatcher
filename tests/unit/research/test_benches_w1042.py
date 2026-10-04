"""Wave-1042 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1042 import (
    bench_food_chemistry_family,
    bench_food_microbiology_family,
    bench_food_processing_family,
    bench_food_safety_family,
    bench_nutrition_science_family,
    bench_sensory_evaluation_family,
)

_FAMILY_BENCHES = [
    bench_food_chemistry_family,
    bench_food_microbiology_family,
    bench_food_processing_family,
    bench_nutrition_science_family,
    bench_sensory_evaluation_family,
    bench_food_safety_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
