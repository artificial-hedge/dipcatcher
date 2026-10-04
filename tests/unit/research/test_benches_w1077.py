"""Wave-1077 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1077 import (
    bench_art_conservation_family,
    bench_art_history_family,
    bench_painting_techniques_family,
    bench_printmaking_family,
    bench_sculpture_methods_family,
    bench_visual_culture_family,
)

_FAMILY_BENCHES = [
    bench_painting_techniques_family,
    bench_sculpture_methods_family,
    bench_printmaking_family,
    bench_art_conservation_family,
    bench_art_history_family,
    bench_visual_culture_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
