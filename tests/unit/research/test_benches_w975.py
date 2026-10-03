"""Wave-975 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w975 import (
    bench_dist_convolution_family,
    bench_paley_wiener_family,
    bench_schwartz_dist_family,
    bench_sing_support_family,
    bench_sobolev_trace_family,
    bench_temper_dist_family,
)

_FAMILY_BENCHES = [
    bench_schwartz_dist_family,
    bench_temper_dist_family,
    bench_dist_convolution_family,
    bench_sing_support_family,
    bench_paley_wiener_family,
    bench_sobolev_trace_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
