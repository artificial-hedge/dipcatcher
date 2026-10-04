"""Wave-990 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w990 import (
    bench_euler_lagrange_family,
    bench_geodesic_var_family,
    bench_isoperimetric_var_family,
    bench_jacobi_eq_family,
    bench_legendre_cond_family,
    bench_soap_film_family,
)

_FAMILY_BENCHES = [
    bench_euler_lagrange_family,
    bench_legendre_cond_family,
    bench_jacobi_eq_family,
    bench_geodesic_var_family,
    bench_isoperimetric_var_family,
    bench_soap_film_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
