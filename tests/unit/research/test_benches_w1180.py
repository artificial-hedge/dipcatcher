"""Wave-1180 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1180 import (
    bench_allied_health_family,
    bench_midwifery_family,
    bench_nursing_studies_family,
    bench_occupational_science_family,
    bench_paramedicine_family,
    bench_speech_pathology_family,
)

_FAMILY_BENCHES = [
    bench_nursing_studies_family,
    bench_allied_health_family,
    bench_midwifery_family,
    bench_paramedicine_family,
    bench_occupational_science_family,
    bench_speech_pathology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
