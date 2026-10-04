"""Wave-1044 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1044 import (
    bench_blasting_engineering_family,
    bench_mine_design_family,
    bench_mine_ventilation_family,
    bench_mineral_processing_family,
    bench_ore_reserve_estimation_family,
    bench_rock_mechanics_family,
)

_FAMILY_BENCHES = [
    bench_mine_design_family,
    bench_rock_mechanics_family,
    bench_mineral_processing_family,
    bench_blasting_engineering_family,
    bench_mine_ventilation_family,
    bench_ore_reserve_estimation_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
