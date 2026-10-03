"""Wave-989 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w989 import (
    bench_fbi_transform_family,
    bench_melrose_bdy_family,
    bench_parametrix_family,
    bench_propagation_thm_family,
    bench_sg_calculus_family,
    bench_wave_eq_group_family,
)

_FAMILY_BENCHES = [
    bench_parametrix_family,
    bench_wave_eq_group_family,
    bench_propagation_thm_family,
    bench_melrose_bdy_family,
    bench_fbi_transform_family,
    bench_sg_calculus_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
