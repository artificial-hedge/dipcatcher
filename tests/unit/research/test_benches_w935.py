"""Wave-935 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w935 import (
    bench_analytic_center_family,
    bench_cvx_reform_family,
    bench_dik_ellipsoid_family,
    bench_kkt_solve_family,
    bench_logbarrier_fn_family,
    bench_self_concordant_family,
)

_FAMILY_BENCHES = [
    bench_kkt_solve_family,
    bench_cvx_reform_family,
    bench_self_concordant_family,
    bench_logbarrier_fn_family,
    bench_analytic_center_family,
    bench_dik_ellipsoid_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
