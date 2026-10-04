"""Wave-969 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w969 import (
    bench_bott_periodicity_k_family,
    bench_elliott_invariant_family,
    bench_k0_algebra_family,
    bench_k1_algebra_family,
    bench_pimsner_voicul_family,
    bench_six_term_exact_family,
)

_FAMILY_BENCHES = [
    bench_k0_algebra_family,
    bench_k1_algebra_family,
    bench_bott_periodicity_k_family,
    bench_six_term_exact_family,
    bench_pimsner_voicul_family,
    bench_elliott_invariant_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
