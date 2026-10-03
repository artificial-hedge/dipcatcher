"""Wave-938 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w938 import (
    bench_averaged_operator_family,
    bench_cocoercive_family,
    bench_fejer_monotone_family,
    bench_firmly_nonexpansive_family,
    bench_monotone_inclusion_family,
    bench_quasinonexpansive_family,
)

_FAMILY_BENCHES = [
    bench_fejer_monotone_family,
    bench_firmly_nonexpansive_family,
    bench_averaged_operator_family,
    bench_cocoercive_family,
    bench_quasinonexpansive_family,
    bench_monotone_inclusion_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
