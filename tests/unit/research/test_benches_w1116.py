"""Wave-1116 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1116 import (
    bench_comparative_psychology_family,
    bench_environmental_psychology_family,
    bench_evolutionary_psychology_family,
    bench_experimental_psychology_family,
    bench_psychopathology_family,
    bench_sport_psychology_family,
)

_FAMILY_BENCHES = [
    bench_experimental_psychology_family,
    bench_comparative_psychology_family,
    bench_evolutionary_psychology_family,
    bench_psychopathology_family,
    bench_environmental_psychology_family,
    bench_sport_psychology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
