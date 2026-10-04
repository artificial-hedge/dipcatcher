"""Wave-1089 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1089 import (
    bench_computational_linguistics_family,
    bench_corpus_linguistics_family,
    bench_dialectology_family,
    bench_historical_linguistics_family,
    bench_psycholinguistics_family,
    bench_sociolinguistics_family,
)

_FAMILY_BENCHES = [
    bench_sociolinguistics_family,
    bench_psycholinguistics_family,
    bench_computational_linguistics_family,
    bench_corpus_linguistics_family,
    bench_dialectology_family,
    bench_historical_linguistics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
