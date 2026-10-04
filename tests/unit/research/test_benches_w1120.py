"""Wave-1120 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1120 import (
    bench_applied_anthropology_family,
    bench_digital_anthropology_family,
    bench_environmental_anthropology_family,
    bench_forensic_anthropology_family,
    bench_psychological_anthropology_family,
    bench_visual_anthropology_family,
)

_FAMILY_BENCHES = [
    bench_visual_anthropology_family,
    bench_applied_anthropology_family,
    bench_forensic_anthropology_family,
    bench_digital_anthropology_family,
    bench_environmental_anthropology_family,
    bench_psychological_anthropology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
