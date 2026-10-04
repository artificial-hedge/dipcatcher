"""Wave-1175 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1175 import (
    bench_automotive_technology_family,
    bench_carpentry_trades_family,
    bench_electrical_trades_family,
    bench_plumbing_hvac_family,
    bench_refrigeration_technology_family,
    bench_welding_technology_family,
)

_FAMILY_BENCHES = [
    bench_electrical_trades_family,
    bench_plumbing_hvac_family,
    bench_welding_technology_family,
    bench_carpentry_trades_family,
    bench_automotive_technology_family,
    bench_refrigeration_technology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
