"""Wave-1207 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1207 import (
    bench_art_therapy_family,
    bench_behavioral_therapy_cognitive_family,
    bench_music_therapy_family,
    bench_play_therapy_family,
    bench_psychoanalysis_studies_family,
    bench_psychotherapy_studies_family,
)

_FAMILY_BENCHES = [
    bench_psychoanalysis_studies_family,
    bench_psychotherapy_studies_family,
    bench_behavioral_therapy_cognitive_family,
    bench_art_therapy_family,
    bench_music_therapy_family,
    bench_play_therapy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
