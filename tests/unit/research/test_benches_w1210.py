"""Wave-1210 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1210 import (
    bench_anxiety_disorders_family,
    bench_forensic_psychiatry_family,
    bench_geriatric_psychiatry_family,
    bench_mood_disorders_family,
    bench_personality_disorders_family,
    bench_psychotic_disorders_family,
)

_FAMILY_BENCHES = [
    bench_forensic_psychiatry_family,
    bench_geriatric_psychiatry_family,
    bench_mood_disorders_family,
    bench_psychotic_disorders_family,
    bench_personality_disorders_family,
    bench_anxiety_disorders_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
