"""Wave-1110 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1110 import (
    bench_phenomenology_2_family,
    bench_philosophy_of_biology_family,
    bench_philosophy_of_history_family,
    bench_philosophy_of_mathematics_family,
    bench_philosophy_of_religion_family,
    bench_process_philosophy_family,
)

_FAMILY_BENCHES = [
    bench_philosophy_of_biology_family,
    bench_philosophy_of_mathematics_family,
    bench_philosophy_of_religion_family,
    bench_phenomenology_2_family,
    bench_philosophy_of_history_family,
    bench_process_philosophy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
