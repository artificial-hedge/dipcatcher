"""Wave-1156 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1156 import (
    bench_applied_mathematics_family,
    bench_bioinformatics_5_family,
    bench_computational_science_family,
    bench_data_science_family,
    bench_probability_4_family,
    bench_statistics_2_family,
)

_FAMILY_BENCHES = [
    bench_applied_mathematics_family,
    bench_statistics_2_family,
    bench_probability_4_family,
    bench_computational_science_family,
    bench_data_science_family,
    bench_bioinformatics_5_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
