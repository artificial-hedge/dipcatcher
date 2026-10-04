"""Wave-911 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w911 import (
    bench_bezier_eval_family,
    bench_chan_hull_family,
    bench_cohen_sutherland_family,
    bench_gift_wrap_family,
    bench_liang_barsky_family,
    bench_monotone_chain_family,
)

_FAMILY_BENCHES = [
    bench_monotone_chain_family,
    bench_gift_wrap_family,
    bench_chan_hull_family,
    bench_liang_barsky_family,
    bench_cohen_sutherland_family,
    bench_bezier_eval_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
