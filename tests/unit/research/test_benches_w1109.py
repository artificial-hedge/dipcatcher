"""Wave-1109 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1109 import (
    bench_anthropological_linguistics_family,
    bench_applied_linguistics_family,
    bench_discourse_analysis_family,
    bench_evolutionary_linguistics_family,
    bench_forensic_linguistics_family,
    bench_neurolinguistics_family,
)

_FAMILY_BENCHES = [
    bench_applied_linguistics_family,
    bench_anthropological_linguistics_family,
    bench_neurolinguistics_family,
    bench_evolutionary_linguistics_family,
    bench_forensic_linguistics_family,
    bench_discourse_analysis_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
