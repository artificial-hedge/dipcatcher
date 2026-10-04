"""Wave-1185 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1185 import (
    bench_esports_studies_family,
    bench_game_design_family,
    bench_game_development_family,
    bench_game_studies_family,
    bench_interactive_media_family,
    bench_ludology_family,
)

_FAMILY_BENCHES = [
    bench_game_design_family,
    bench_esports_studies_family,
    bench_interactive_media_family,
    bench_game_studies_family,
    bench_ludology_family,
    bench_game_development_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
