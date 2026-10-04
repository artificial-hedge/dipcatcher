"""Wave-987 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w987 import (
    bench_davies_gaffney_family,
    bench_gaussian_upper_family,
    bench_grad_est_family,
    bench_li_yau_family,
    bench_nash_ineq_family,
    bench_parabolic_harnack_family,
)

_FAMILY_BENCHES = [
    bench_parabolic_harnack_family,
    bench_gaussian_upper_family,
    bench_li_yau_family,
    bench_nash_ineq_family,
    bench_davies_gaffney_family,
    bench_grad_est_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
