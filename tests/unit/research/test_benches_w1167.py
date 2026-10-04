"""Wave-1167 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1167 import (
    bench_art_history_2_family,
    bench_dance_2_family,
    bench_film_studies_3_family,
    bench_music_2_family,
    bench_performance_studies_2_family,
    bench_theater_2_family,
)

_FAMILY_BENCHES = [
    bench_music_2_family,
    bench_theater_2_family,
    bench_dance_2_family,
    bench_film_studies_3_family,
    bench_art_history_2_family,
    bench_performance_studies_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
