"""Wave-988 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w988 import (
    bench_cheeger_ineq_family,
    bench_heat_invariants_family,
    bench_isospectral_family,
    bench_nodal_domain_family,
    bench_spectral_geometry_family,
    bench_weyl_law_family,
)

_FAMILY_BENCHES = [
    bench_spectral_geometry_family,
    bench_heat_invariants_family,
    bench_weyl_law_family,
    bench_isospectral_family,
    bench_cheeger_ineq_family,
    bench_nodal_domain_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
