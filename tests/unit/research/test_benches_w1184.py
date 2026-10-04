"""Wave-1184 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1184 import (
    bench_animation_studies_family,
    bench_cinematography_studies_family,
    bench_documentary_production_family,
    bench_film_editing_family,
    bench_film_production_family,
    bench_sound_design_family,
)

_FAMILY_BENCHES = [
    bench_film_production_family,
    bench_cinematography_studies_family,
    bench_film_editing_family,
    bench_sound_design_family,
    bench_documentary_production_family,
    bench_animation_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
