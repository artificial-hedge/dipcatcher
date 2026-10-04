"""Wave-1097 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1097 import (
    bench_eastern_philosophy_family,
    bench_moral_philosophy_family,
    bench_philosophy_of_language_family,
    bench_philosophy_of_law_family,
    bench_philosophy_of_mind_family,
    bench_political_philosophy_family,
)

_FAMILY_BENCHES = [
    bench_moral_philosophy_family,
    bench_political_philosophy_family,
    bench_philosophy_of_mind_family,
    bench_philosophy_of_language_family,
    bench_philosophy_of_law_family,
    bench_eastern_philosophy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
