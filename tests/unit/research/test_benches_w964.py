"""Wave-964 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w964 import (
    bench_adjoint_unbounded_family,
    bench_closed_operator_family,
    bench_domain_dense_family,
    bench_resolvent_op_family,
    bench_spectral_measure_family,
    bench_unbounded_operator_family,
)

_FAMILY_BENCHES = [
    bench_unbounded_operator_family,
    bench_closed_operator_family,
    bench_domain_dense_family,
    bench_adjoint_unbounded_family,
    bench_resolvent_op_family,
    bench_spectral_measure_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
