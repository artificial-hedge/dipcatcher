"""Wave-923 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w923 import (
    bench_chinese_restaurant_family,
    bench_dirichlet_process_family,
    bench_hierarchical_dp_family,
    bench_indian_buffet_family,
    bench_pitman_yor_family,
    bench_stick_breaking_family,
)

_FAMILY_BENCHES = [
    bench_dirichlet_process_family,
    bench_stick_breaking_family,
    bench_pitman_yor_family,
    bench_indian_buffet_family,
    bench_chinese_restaurant_family,
    bench_hierarchical_dp_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
