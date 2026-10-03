"""Wave-983 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w983 import (
    bench_atoms_decomp_family,
    bench_besov_embed_family,
    bench_besov_space_family,
    bench_hardy_littlewood_max_family,
    bench_triebel_lizorkin_family,
    bench_wavelet_char_family,
)

_FAMILY_BENCHES = [
    bench_besov_space_family,
    bench_triebel_lizorkin_family,
    bench_atoms_decomp_family,
    bench_wavelet_char_family,
    bench_besov_embed_family,
    bench_hardy_littlewood_max_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
