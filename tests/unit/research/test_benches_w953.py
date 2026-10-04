"""Wave-953 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w953 import (
    bench_bounded_operator_family,
    bench_isometry_operator_family,
    bench_operator_adjoint_family,
    bench_operator_norm_family,
    bench_positive_operator_family,
    bench_projection_operator_family,
)

_FAMILY_BENCHES = [
    bench_bounded_operator_family,
    bench_operator_norm_family,
    bench_operator_adjoint_family,
    bench_projection_operator_family,
    bench_positive_operator_family,
    bench_isometry_operator_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
