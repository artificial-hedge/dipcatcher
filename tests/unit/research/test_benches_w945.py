"""Wave-945 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w945 import (
    bench_hankel_op_family,
    bench_kyfan_norm_family,
    bench_matrix_det_family,
    bench_numerical_radius_family,
    bench_pfaffian_poly_family,
    bench_schatten_norm_family,
)

_FAMILY_BENCHES = [
    bench_kyfan_norm_family,
    bench_schatten_norm_family,
    bench_numerical_radius_family,
    bench_matrix_det_family,
    bench_pfaffian_poly_family,
    bench_hankel_op_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
