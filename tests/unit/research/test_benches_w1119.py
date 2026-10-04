"""Wave-1119 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1119 import (
    bench_cultural_history_family,
    bench_diplomatic_history_family,
    bench_history_of_medicine_family,
    bench_history_of_technology_family,
    bench_military_history_family,
    bench_social_history_family,
)

_FAMILY_BENCHES = [
    bench_social_history_family,
    bench_cultural_history_family,
    bench_military_history_family,
    bench_diplomatic_history_family,
    bench_history_of_technology_family,
    bench_history_of_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
