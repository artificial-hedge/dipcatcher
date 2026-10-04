"""Wave-963 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w963 import (
    bench_atkinson_thm_family,
    bench_browder_operator_family,
    bench_essential_spectrum_family,
    bench_fredholm_index_family,
    bench_riesz_schauder_family,
    bench_weyl_theorem_family,
)

_FAMILY_BENCHES = [
    bench_fredholm_index_family,
    bench_weyl_theorem_family,
    bench_essential_spectrum_family,
    bench_browder_operator_family,
    bench_riesz_schauder_family,
    bench_atkinson_thm_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
