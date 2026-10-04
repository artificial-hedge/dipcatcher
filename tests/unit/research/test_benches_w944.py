"""Wave-944 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w944 import (
    bench_cauchy_binet_family,
    bench_fan_inequality_family,
    bench_horn_inequality_family,
    bench_majorization_vec_family,
    bench_schur_complement_family,
    bench_weyl_ineq_family,
)

_FAMILY_BENCHES = [
    bench_fan_inequality_family,
    bench_horn_inequality_family,
    bench_weyl_ineq_family,
    bench_cauchy_binet_family,
    bench_schur_complement_family,
    bench_majorization_vec_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
