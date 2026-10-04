"""Wave-1021 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1021 import (
    bench_branching_epidemic_family,
    bench_herd_immunity_family,
    bench_r0_estimation_family,
    bench_seir_epidemic_family,
    bench_sir_epidemic_family,
    bench_sis_epidemic_family,
)

_FAMILY_BENCHES = [
    bench_sir_epidemic_family,
    bench_sis_epidemic_family,
    bench_seir_epidemic_family,
    bench_r0_estimation_family,
    bench_herd_immunity_family,
    bench_branching_epidemic_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
