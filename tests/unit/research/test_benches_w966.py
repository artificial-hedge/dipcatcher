"""Wave-966 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w966 import (
    bench_cb_map_family,
    bench_complete_contraction_family,
    bench_injective_space_family,
    bench_noncommutative_lp_family,
    bench_oh_emb_family,
    bench_operator_space_family,
)

_FAMILY_BENCHES = [
    bench_operator_space_family,
    bench_cb_map_family,
    bench_complete_contraction_family,
    bench_injective_space_family,
    bench_noncommutative_lp_family,
    bench_oh_emb_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
