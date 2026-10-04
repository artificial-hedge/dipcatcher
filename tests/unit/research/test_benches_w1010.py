"""Wave-1010 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1010 import (
    bench_canonical_quantization_family,
    bench_dirac_equation_family,
    bench_feynman_rules_family,
    bench_klein_gordon_family,
    bench_path_integral_qm_family,
    bench_renormalization_group_family,
)

_FAMILY_BENCHES = [
    bench_klein_gordon_family,
    bench_dirac_equation_family,
    bench_feynman_rules_family,
    bench_renormalization_group_family,
    bench_path_integral_qm_family,
    bench_canonical_quantization_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
