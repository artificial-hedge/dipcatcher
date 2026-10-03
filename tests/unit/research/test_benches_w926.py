"""Wave-926 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w926 import (
    bench_alpha_divergence_family,
    bench_amari_connection_family,
    bench_csiszar_div_family,
    bench_dual_connection_family,
    bench_f_divergence_family,
    bench_tsallis_entropy_family,
)

_FAMILY_BENCHES = [
    bench_f_divergence_family,
    bench_alpha_divergence_family,
    bench_csiszar_div_family,
    bench_amari_connection_family,
    bench_dual_connection_family,
    bench_tsallis_entropy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
