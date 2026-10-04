"""Wave-1190 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1190 import (
    bench_graphic_design_family,
    bench_motion_graphics_family,
    bench_photography_studies_family,
    bench_print_media_family,
    bench_typography_studies_family,
    bench_web_design_family,
)

_FAMILY_BENCHES = [
    bench_graphic_design_family,
    bench_typography_studies_family,
    bench_photography_studies_family,
    bench_print_media_family,
    bench_web_design_family,
    bench_motion_graphics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
