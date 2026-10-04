"""Wave-973 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w973 import (
    bench_banach_mazur_family,
    bench_djt_space_family,
    bench_gl_property_family,
    bench_kalton_loc_family,
    bench_schauder_basis_family,
    bench_type_cotype_family,
)

_FAMILY_BENCHES = [
    bench_banach_mazur_family,
    bench_type_cotype_family,
    bench_gl_property_family,
    bench_djt_space_family,
    bench_schauder_basis_family,
    bench_kalton_loc_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
