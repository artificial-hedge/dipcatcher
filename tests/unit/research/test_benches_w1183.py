"""Wave-1183 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1183 import (
    bench_composition_studies_family,
    bench_ethnomusicology_2_family,
    bench_music_cognition_2_family,
    bench_music_theory_2_family,
    bench_musicology_2_family,
    bench_organology_2_family,
)

_FAMILY_BENCHES = [
    bench_music_theory_2_family,
    bench_musicology_2_family,
    bench_ethnomusicology_2_family,
    bench_music_cognition_2_family,
    bench_organology_2_family,
    bench_composition_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
