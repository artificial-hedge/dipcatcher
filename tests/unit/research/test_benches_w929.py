"""Wave-929 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w929 import (
    bench_bhat_distance_family,
    bench_chi_square_div_family,
    bench_d_total_var_family,
    bench_hellinger_dist_family,
    bench_jeffreys_div_family,
    bench_mahalanobis_div_family,
)

_FAMILY_BENCHES = [
    bench_mahalanobis_div_family,
    bench_bhat_distance_family,
    bench_hellinger_dist_family,
    bench_jeffreys_div_family,
    bench_d_total_var_family,
    bench_chi_square_div_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
