"""Wave-1132 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1132 import (
    bench_history_of_capitalism_family,
    bench_history_of_emotions_family,
    bench_history_of_religions_family,
    bench_history_of_sexuality_family,
    bench_history_of_the_book_family,
    bench_microhistory_family,
)

_FAMILY_BENCHES = [
    bench_history_of_emotions_family,
    bench_history_of_sexuality_family,
    bench_history_of_the_book_family,
    bench_history_of_capitalism_family,
    bench_history_of_religions_family,
    bench_microhistory_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
