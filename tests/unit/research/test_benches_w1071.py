"""Wave-1071 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1071 import (
    bench_ethnomusicology_family,
    bench_music_cognition_family,
    bench_music_history_family,
    bench_music_theory_family,
    bench_musicology_family,
    bench_organology_family,
)

_FAMILY_BENCHES = [
    bench_musicology_family,
    bench_ethnomusicology_family,
    bench_music_theory_family,
    bench_music_cognition_family,
    bench_organology_family,
    bench_music_history_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
