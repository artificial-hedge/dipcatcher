"""Wave-914 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w914 import (
    bench_green_function_bvp_family,
    bench_invariant_imbedding_family,
    bench_ralston_rk_family,
    bench_ralston_second_family,
    bench_runge_kutta4_family,
    bench_verner_rk_family,
)

_FAMILY_BENCHES = [
    bench_ralston_rk_family,
    bench_verner_rk_family,
    bench_ralston_second_family,
    bench_runge_kutta4_family,
    bench_invariant_imbedding_family,
    bench_green_function_bvp_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
