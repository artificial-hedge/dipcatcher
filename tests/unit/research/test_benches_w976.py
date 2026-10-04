"""Wave-976 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w976 import (
    bench_diam_dim_family,
    bench_frechet_nuclear_family,
    bench_gelfand_triple_family,
    bench_hilbert_schmidt_emb_family,
    bench_nuclear_map_family,
    bench_trace_duality_family,
)

_FAMILY_BENCHES = [
    bench_nuclear_map_family,
    bench_frechet_nuclear_family,
    bench_gelfand_triple_family,
    bench_hilbert_schmidt_emb_family,
    bench_trace_duality_family,
    bench_diam_dim_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
