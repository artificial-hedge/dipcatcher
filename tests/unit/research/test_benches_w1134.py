"""Wave-1134 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1134 import (
    bench_adaptive_method_theory_family,
    bench_finite_element_theory_family,
    bench_high_performance_numerics_family,
    bench_reduced_order_modeling_family,
    bench_spectral_theory_numerics_family,
    bench_uncertainty_quantification_2_family,
)

_FAMILY_BENCHES = [
    bench_finite_element_theory_family,
    bench_spectral_theory_numerics_family,
    bench_adaptive_method_theory_family,
    bench_reduced_order_modeling_family,
    bench_uncertainty_quantification_2_family,
    bench_high_performance_numerics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
