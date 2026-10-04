"""Wave-1124 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1124 import (
    bench_sociology_of_aging_family,
    bench_sociology_of_emotions_family,
    bench_sociology_of_food_family,
    bench_sociology_of_media_family,
    bench_sociology_of_sport_family,
    bench_sociology_of_work_family,
)

_FAMILY_BENCHES = [
    bench_sociology_of_work_family,
    bench_sociology_of_emotions_family,
    bench_sociology_of_food_family,
    bench_sociology_of_media_family,
    bench_sociology_of_sport_family,
    bench_sociology_of_aging_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
