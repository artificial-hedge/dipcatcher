"""Wave-1034 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1034 import (
    bench_ergonomics_family,
    bench_facility_layout_family,
    bench_manufacturing_sys_family,
    bench_operations_research_family,
    bench_quality_control_family,
    bench_supply_chain_family,
)

_FAMILY_BENCHES = [
    bench_operations_research_family,
    bench_supply_chain_family,
    bench_manufacturing_sys_family,
    bench_quality_control_family,
    bench_ergonomics_family,
    bench_facility_layout_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
