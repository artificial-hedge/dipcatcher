"""Wave-1030 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1030 import (
    bench_circuit_analysis_family,
    bench_control_systems_family,
    bench_electromagnetics_family,
    bench_power_systems_family,
    bench_semiconductor_family,
    bench_signal_processing2_family,
)

_FAMILY_BENCHES = [
    bench_circuit_analysis_family,
    bench_power_systems_family,
    bench_control_systems_family,
    bench_signal_processing2_family,
    bench_electromagnetics_family,
    bench_semiconductor_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
