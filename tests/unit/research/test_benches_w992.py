"""Wave-992 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w992 import (
    bench_limiting_absorption_family,
    bench_radiation_cond_family,
    bench_resonances_thy_family,
    bench_scattering_matrix_family,
    bench_trace_class_scatt_family,
    bench_wave_operators_family,
)

_FAMILY_BENCHES = [
    bench_wave_operators_family,
    bench_scattering_matrix_family,
    bench_limiting_absorption_family,
    bench_trace_class_scatt_family,
    bench_resonances_thy_family,
    bench_radiation_cond_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
