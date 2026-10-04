"""Wave-934 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w934 import (
    bench_inf_convolution_family,
    bench_legendre_transform_family,
    bench_normal_cone_family,
    bench_perspective_fn_family,
    bench_polar_cone_family,
    bench_support_fn_family,
)

_FAMILY_BENCHES = [
    bench_inf_convolution_family,
    bench_legendre_transform_family,
    bench_support_fn_family,
    bench_perspective_fn_family,
    bench_polar_cone_family,
    bench_normal_cone_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
