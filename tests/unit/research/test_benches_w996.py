"""Wave-996 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w996 import (
    bench_dbar_method_family,
    bench_deift_zhou_family,
    bench_fokas_unified_family,
    bench_isomonodromy_family,
    bench_orthogonal_poly_rh_family,
    bench_small_norm_rh_family,
)

_FAMILY_BENCHES = [
    bench_dbar_method_family,
    bench_orthogonal_poly_rh_family,
    bench_isomonodromy_family,
    bench_fokas_unified_family,
    bench_deift_zhou_family,
    bench_small_norm_rh_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
