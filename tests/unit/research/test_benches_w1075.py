"""Wave-1075 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1075 import (
    bench_architecture_theory_family,
    bench_building_science_family,
    bench_industrial_design_family,
    bench_interior_design_family,
    bench_landscape_architecture_family,
    bench_urban_design_family,
)

_FAMILY_BENCHES = [
    bench_architecture_theory_family,
    bench_urban_design_family,
    bench_landscape_architecture_family,
    bench_interior_design_family,
    bench_industrial_design_family,
    bench_building_science_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
