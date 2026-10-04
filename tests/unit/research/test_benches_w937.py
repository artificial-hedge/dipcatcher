"""Wave-937 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w937 import (
    bench_chambolle_pock_family,
    bench_davis_yin_family,
    bench_douglas_rachford_family,
    bench_forward_backward_family,
    bench_peaceman_rachford_family,
    bench_tseng_split_family,
)

_FAMILY_BENCHES = [
    bench_douglas_rachford_family,
    bench_peaceman_rachford_family,
    bench_tseng_split_family,
    bench_forward_backward_family,
    bench_chambolle_pock_family,
    bench_davis_yin_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
